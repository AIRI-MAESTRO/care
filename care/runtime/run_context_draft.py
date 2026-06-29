"""Run-context-modal data layer (TODO §1.3 P1).

The `RunContextModal` in `LibraryScreen` opens when the user picks
``Run`` on a saved agent. The modal lets them:

* (a) reuse the original task description + context files captured
  at generation time,
* (b) edit the task description,
* (c) swap out individual context files (adds / drops / replaces),
* (d) override the LLM model / base_url per run.

The Textual modal itself is gated on TODO §1 P0 multi-screen
workflow, but the form-binding model + validation + projection
into `CareConfig` overrides + handoff into
:func:`care.runtime.prime_from_saved_chain` are all bounded
concerns that ship now as the data layer.

What this module provides:

* :class:`ContextFile` — frozen file ref the modal binds one row
  per. Mirrors the SDK's :class:`ContextFileRef` but adds a
  ``status`` field (``"saved"`` / ``"added"`` / ``"replaced"`` /
  ``"missing"``) so the modal can colour-code rows.
* :class:`RunContextDraft` — frozen form state the modal updates.
  Carries the original chain inputs plus any user edits, with
  ``has_edits`` / ``has_overrides`` predicates the modal uses to
  toggle the "Run" / "Run (modified)" button label.
* :func:`extract_run_context_draft` — pure projection from a saved
  `ReasoningChain` (or a duck-typed object exposing
  ``get_care_metadata()`` / ``entity_id``) into the form's initial
  state.
* :func:`validate_run_context_draft` — surfaces a list of
  :class:`RunContextIssue` rows the modal renders inline next to
  problematic fields (empty task, missing file, etc.).
* :func:`apply_overrides` — projects the draft's per-run model /
  provider overrides onto a `CareConfig` clone, leaving the
  caller's session config untouched.
* :func:`build_extra_kwargs` — bundles the modal's
  edited values into the ``files=`` / ``outer_context=`` /
  ``load_files_from_metadata=`` kwargs
  :func:`care.runtime.prime_from_saved_chain` already accepts.
"""

from __future__ import annotations

import hashlib
import mimetypes
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Optional

from care.runtime.file_loading import (
    MAX_CONTEXT_FILE_CHARS,
    load_file,
    load_file_text,
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


FileStatus = Literal["saved", "added", "replaced", "dropped", "missing"]
"""Per-row state the modal renders against:

* ``saved`` — file was present at generation time and the user
  hasn't touched it.
* ``added`` — user added a new file on top of the saved set.
* ``replaced`` — user swapped the file at the same logical key
  (path stays, sha256 / size may have changed).
* ``dropped`` — saved file the user removed from the run.
  Kept in the draft so the modal can offer "undo".
* ``missing`` — saved file path no longer exists on disk.
"""


@dataclass(frozen=True)
class ContextFile:
    """One row in the modal's file list."""

    path: str
    sha256: str = ""
    size_bytes: int = 0
    mime_type: Optional[str] = None
    status: FileStatus = "saved"

    @property
    def is_active(self) -> bool:
        """``True`` when this file is included in the upcoming run.
        ``dropped`` files are kept on the draft for undo affordance
        but excluded from the actual ``files=`` payload."""
        return self.status != "dropped"

    @property
    def is_user_edit(self) -> bool:
        """Did the user touch this row vs. inheriting it from the
        saved chain? Drives the "modified" button label."""
        return self.status in ("added", "replaced", "dropped")


@dataclass(frozen=True)
class RunContextDraft:
    """Form state for the RunContextModal.

    Frozen — the modal builds a fresh instance via
    :func:`dataclasses.replace` on every field edit. Lets the
    snapshot flow through Textual messages without defensive
    copies and keeps the undo stack trivial (a list of past
    draft instances).
    """

    source_entity_id: str
    source_name: str = ""
    original_task: str = ""
    task_description: str = ""
    files: tuple[ContextFile, ...] = ()
    model_override: Optional[str] = None
    base_url_override: Optional[str] = None
    api_key_override: Optional[str] = None
    streaming_enabled: bool = True
    extra_files: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def task_edited(self) -> bool:
        """``True`` when the user edited the task away from the
        chain's original ``task_description``."""
        return self.task_description.strip() != self.original_task.strip()

    @property
    def has_overrides(self) -> bool:
        """``True`` when any model / base_url / api_key override is set."""
        return (
            self.model_override is not None
            or self.base_url_override is not None
            or self.api_key_override is not None
        )

    @property
    def has_file_edits(self) -> bool:
        return any(f.is_user_edit for f in self.files) or bool(self.extra_files)

    @property
    def has_edits(self) -> bool:
        """Drives the "Run" / "Run (modified)" button label."""
        return self.task_edited or self.has_overrides or self.has_file_edits

    @property
    def active_files(self) -> tuple[ContextFile, ...]:
        """Files the upcoming run will see (drops excluded)."""
        return tuple(f for f in self.files if f.is_active)


@dataclass(frozen=True)
class RunContextIssue:
    """One validation finding the modal renders inline."""

    severity: Literal["error", "warning"]
    field: Literal["task_description", "files", "model_override", "base_url_override"]
    message: str
    detail: Optional[str] = None


# ---------------------------------------------------------------------------
# Projection from a saved chain
# ---------------------------------------------------------------------------


def extract_run_context_draft(
    chain: Any,
    *,
    source_name: str = "",
) -> RunContextDraft:
    """Project a saved `ReasoningChain` into the modal's initial
    draft state.

    Accepts:
        * A `ReasoningChain` with `get_care_metadata()` (PREPARE.md
          §5.6 — CARL ships this on every chain CARE saved).
        * A duck-typed object exposing `metadata` (dict) plus
          ``entity_id`` / ``id``.
        * A plain dict matching the same shape (used in tests).

    Args:
        chain: Saved chain object.
        source_name: Human-readable label for the modal header
            (typically the agent's display name). Falls back to
            the chain's `display_name` from CARE metadata when
            empty.

    Returns:
        Initial :class:`RunContextDraft`. The modal mutates it via
        :func:`dataclasses.replace` for every field edit.
    """
    metadata = _extract_care_metadata(chain)
    entity_id = _read(chain, "entity_id") or _read(chain, "id") or ""

    task = ""
    files: tuple[ContextFile, ...] = ()
    label = source_name

    if metadata:
        raw_task = metadata.get("task_description")
        if isinstance(raw_task, str):
            task = raw_task
        raw_files = metadata.get("context_files") or []
        files = tuple(_project_file(f) for f in raw_files if f)
        if not label:
            disp = metadata.get("display_name")
            if isinstance(disp, str):
                label = disp

    return RunContextDraft(
        source_entity_id=str(entity_id),
        source_name=label,
        original_task=task,
        task_description=task,
        files=files,
        metadata=dict(metadata) if metadata else {},
    )


def _extract_care_metadata(chain: Any) -> dict[str, Any]:
    """Read the CARE metadata block whichever way the chain
    exposes it."""
    # Preferred: typed accessor shipped on CARL's ReasoningChain.
    getter = getattr(chain, "get_care_metadata", None)
    if callable(getter):
        try:
            value = getter()
        except Exception:  # noqa: BLE001
            value = None
        # ``None`` means "no `care` namespace on this chain" — CARL's own
        # contract says callers should then fall back to the raw
        # ``chain.metadata`` (CARE's facade stamps task_description & co.
        # FLAT under metadata, not nested under metadata["care"]). Don't
        # return early, or every facade-saved chain loses its task.
        if value is not None:
            model_dump = getattr(value, "model_dump", None)
            if callable(model_dump):
                try:
                    dumped = model_dump(exclude_none=False)
                except TypeError:
                    dumped = model_dump()
                if isinstance(dumped, dict):
                    return dict(dumped)
            if isinstance(value, dict):
                return dict(value)
    # Fallback: read the raw metadata dict.
    raw = _read(chain, "metadata")
    if isinstance(raw, dict):
        care_block = raw.get("care") or raw.get("metadata") or raw
        if isinstance(care_block, dict):
            return dict(care_block)
    return {}


def _project_file(raw: Any) -> ContextFile:
    """Map either an SDK `ContextFileRef` or a plain dict into
    :class:`ContextFile` with ``status="saved"``."""
    if isinstance(raw, dict):
        path = str(raw.get("path") or "")
        sha = str(raw.get("sha256") or "")
        size = int(raw.get("size_bytes") or 0)
        mime = raw.get("mime_type")
    else:
        path = str(getattr(raw, "path", "") or "")
        sha = str(getattr(raw, "sha256", "") or "")
        size = int(getattr(raw, "size_bytes", 0) or 0)
        mime = getattr(raw, "mime_type", None)
    return ContextFile(
        path=path,
        sha256=sha,
        size_bytes=size,
        mime_type=mime if isinstance(mime, str) else None,
        status="saved",
    )


# ---------------------------------------------------------------------------
# Mutators
# ---------------------------------------------------------------------------


def set_task(draft: RunContextDraft, task: str) -> RunContextDraft:
    """Return a new draft with ``task_description`` set to
    ``task``. Convenience wrapper around `dataclasses.replace` so
    the modal doesn't import dataclasses directly."""
    return replace(draft, task_description=task)


def add_file(
    draft: RunContextDraft,
    path: str,
    *,
    sha256: str = "",
    size_bytes: int = 0,
    mime_type: Optional[str] = None,
) -> RunContextDraft:
    """Append a new context file to the draft with
    ``status="added"``."""
    new_file = ContextFile(
        path=path,
        sha256=sha256,
        size_bytes=size_bytes,
        mime_type=mime_type,
        status="added",
    )
    return replace(draft, files=draft.files + (new_file,))


def resolve_file_arg(raw: str) -> str:
    """Normalise a user-typed file reference into an absolute path.

    Mirrors the chat surface's ``@`` contract enough for the
    RunContextModal's quick-attach input: a leading ``@`` is
    stripped, surrounding quotes are honoured (``@"my notes.md"``),
    ``~`` is expanded, and a relative path is resolved against the
    current working directory. Returns ``""`` for blank input.

    Existence is *not* checked here — the caller decides whether a
    non-file path is an error (the modal flashes a hint) so this
    stays a pure string transform that's trivial to unit-test.
    """
    s = (raw or "").strip()
    if s.startswith("@"):
        s = s[1:].strip()
    if len(s) >= 2 and s[0] in "\"'" and s[-1] == s[0]:
        s = s[1:-1]
    if not s:
        return ""
    p = Path(s).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    return os.path.normpath(str(p))


def compute_file_stat(path: str) -> tuple[str, int, Optional[str]]:
    """Best-effort ``(sha256_hex, size_bytes, mime_type)`` for a file.

    Used when the modal attaches a freshly-browsed file so the new
    :class:`ContextFile` carries the same shape the saved-chain
    metadata stores (and so a later re-save round-trips a valid
    :class:`gigaevo_client.ContextFileRef`). Returns
    ``("", 0, None)`` when the file can't be read — the attach still
    proceeds; the path alone is enough for the executor to load it.
    """
    try:
        hasher = hashlib.sha256()
        size = 0
        with open(Path(path).expanduser(), "rb") as fp:
            for chunk in iter(lambda: fp.read(65536), b""):
                hasher.update(chunk)
                size += len(chunk)
    except OSError:
        return "", 0, None
    mime, _ = mimetypes.guess_type(str(path))
    return hasher.hexdigest(), size, mime


def attach_path(draft: RunContextDraft, raw_path: str) -> RunContextDraft:
    """Resolve ``raw_path`` and append it to the draft as an
    ``"added"`` file, computing its sha256 / size / mime from disk.

    De-dupes against the current set: a path that's already active
    is a no-op; a path that was previously *dropped* is restored
    rather than duplicated. Returns the draft unchanged when the
    reference resolves to an empty string.
    """
    resolved = resolve_file_arg(raw_path)
    if not resolved:
        return draft
    for f in draft.files:
        if f.path == resolved:
            if f.status == "dropped":
                return restore_file(draft, resolved)
            return draft  # already present + active — no duplicate row
    sha, size, mime = compute_file_stat(resolved)
    return add_file(
        draft, resolved, sha256=sha, size_bytes=size, mime_type=mime,
    )


def drop_file(draft: RunContextDraft, path: str) -> RunContextDraft:
    """Mark the file at ``path`` as ``dropped``. Saved files stay
    in the tuple (so the modal can offer undo) but are excluded
    from :attr:`active_files`. Added files vanish outright."""
    updated: list[ContextFile] = []
    for f in draft.files:
        if f.path != path:
            updated.append(f)
            continue
        if f.status == "added":
            continue  # Truly remove user-added rows.
        updated.append(replace(f, status="dropped"))
    return replace(draft, files=tuple(updated))


def restore_file(draft: RunContextDraft, path: str) -> RunContextDraft:
    """Undo :func:`drop_file` for a dropped saved file. No-op for
    paths that aren't currently dropped."""
    updated: list[ContextFile] = []
    for f in draft.files:
        if f.path == path and f.status == "dropped":
            updated.append(replace(f, status="saved"))
        else:
            updated.append(f)
    return replace(draft, files=tuple(updated))


def replace_file(
    draft: RunContextDraft,
    path: str,
    *,
    sha256: str,
    size_bytes: int,
    mime_type: Optional[str] = None,
) -> RunContextDraft:
    """Swap the sha256 / size / mime on an existing file (same
    path). The status flips to ``replaced`` so the modal renders
    the row as edited."""
    updated: list[ContextFile] = []
    seen = False
    for f in draft.files:
        if f.path == path:
            updated.append(
                replace(
                    f,
                    sha256=sha256,
                    size_bytes=size_bytes,
                    mime_type=mime_type,
                    status="replaced",
                )
            )
            seen = True
        else:
            updated.append(f)
    if not seen:
        # Replacing a non-existent path → no-op; the modal should
        # call `add_file` instead. Returning the draft unchanged
        # keeps the API forgiving.
        return draft
    return replace(draft, files=tuple(updated))


def set_model_override(
    draft: RunContextDraft,
    *,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> RunContextDraft:
    """Set or clear the per-run model / base_url / api_key overrides.

    Passing ``None`` for a field clears it (returns to the
    config default).
    """
    return replace(
        draft,
        model_override=model,
        base_url_override=base_url,
        api_key_override=api_key,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_run_context_draft(
    draft: RunContextDraft,
    *,
    check_files: bool = True,
) -> tuple[RunContextIssue, ...]:
    """Check the draft and return any issues the modal should
    render inline.

    Args:
        draft: Current form state.
        check_files: Stat every active file's ``path`` to detect
            ``"missing"``. ``False`` skips disk I/O — useful when
            tests don't have a real workspace.

    Returns:
        Tuple of issues; empty when the draft is OK to submit.
    """
    issues: list[RunContextIssue] = []
    if not draft.task_description.strip():
        issues.append(
            RunContextIssue(
                severity="error",
                field="task_description",
                message="Task description is required",
            )
        )

    if check_files:
        for f in draft.active_files:
            if not f.path:
                continue
            try:
                exists = Path(f.path).expanduser().exists()
            except OSError:
                exists = False
            if not exists:
                issues.append(
                    RunContextIssue(
                        severity="warning",
                        field="files",
                        message=f"File not found: {f.path}",
                        detail=f.path,
                    )
                )

    if draft.model_override is not None and not draft.model_override.strip():
        issues.append(
            RunContextIssue(
                severity="error",
                field="model_override",
                message="Model override cannot be blank — clear the field to use the default",
            )
        )

    if draft.base_url_override is not None and not draft.base_url_override.strip():
        issues.append(
            RunContextIssue(
                severity="error",
                field="base_url_override",
                message="Provider override cannot be blank — clear the field to use the default",
            )
        )

    return tuple(issues)


def missing_active_files(draft: RunContextDraft) -> tuple[ContextFile, ...]:
    """Return the active context files whose ``path`` no longer
    resolves on disk.

    These are the inputs the chain *expects* (it was generated /
    last run with them) but that aren't currently available — the
    exact set that, left unaddressed, makes CARL prime the run with
    a ``"[missing context file: …]"`` placeholder instead of the
    real contents. The RunContextModal surfaces them as a banner so
    the user attaches a replacement (Browse / ``@path``) or
    explicitly drops the row before running.
    """
    out: list[ContextFile] = []
    for f in draft.active_files:
        if not f.path:
            continue
        try:
            exists = Path(f.path).expanduser().is_file()
        except OSError:
            exists = False
        if not exists:
            out.append(f)
    return tuple(out)


# ---------------------------------------------------------------------------
# Projection back to runtime
# ---------------------------------------------------------------------------


def apply_overrides(config: Any, draft: RunContextDraft) -> Any:
    """Project the draft's per-run model / base_url overrides
    onto a deep copy of ``config``.

    The caller's session-wide config stays untouched so a single
    re-run with a one-off model doesn't pollute the rest of the
    session.

    Args:
        config: A `CareConfig`-like object whose `.mage` block
            carries `model` / `provider` fields.
        draft: Current form state.

    Returns:
        A new config object with overrides applied. Returns the
        original config unchanged when no overrides are set.
    """
    if not draft.has_overrides:
        return config
    if not hasattr(config, "model_copy"):
        # Plain dict / unfamiliar object — best-effort. Return as
        # is; the modal can render a banner.
        return config
    updates: dict[str, Any] = {}
    mage_updates: dict[str, Any] = {}
    if draft.model_override is not None:
        mage_updates["model"] = draft.model_override
    if draft.base_url_override is not None:
        mage_updates["base_url"] = draft.base_url_override
    if draft.api_key_override is not None:
        mage_updates["api_key"] = draft.api_key_override
    if mage_updates and hasattr(config.mage, "model_copy"):
        updates["mage"] = config.mage.model_copy(update=mage_updates)
    return config.model_copy(update=updates) if updates else config


def build_extra_kwargs(draft: RunContextDraft) -> dict[str, Any]:
    """Translate the draft into kwargs for
    :func:`care.runtime.prime_from_saved_chain`.

    The pre-shipped helper already supports ``outer_context=`` +
    ``files=`` + ``load_files_from_metadata=`` — this function
    bundles the modal's edits into that shape so the screen has
    one call site:

    .. code-block:: python

        kwargs = build_extra_kwargs(draft)
        context = prime_from_saved_chain(chain, api=api, **kwargs)

    Returns a dict containing ONLY the keys the modal touched
    (others stay at the helper's defaults) — keeps the call
    explicit when nothing's been edited.
    """
    out: dict[str, Any] = {}
    active_paths = tuple(f.path for f in draft.active_files if f.path)
    saved_paths = tuple(
        f.path for f in draft.files if f.status == "saved" and f.path
    )
    files_edited = draft.has_file_edits or active_paths != saved_paths

    # Read each active file ONCE via the canonical loader — binary-aware
    # (office/pdf → text, images → data URI), size-capped. Only needed when
    # the file set changed (the no-edit path lets CARL auto-load the saved
    # files from metadata).
    reads = {
        cf.path: load_file(cf.path)
        for cf in draft.active_files
        if cf.path
    } if files_edited else {}

    # --- outer_context: inject the attached file CONTENT so the chain
    # actually sees it. A generic chain (e.g. a news agent) never reads
    # `$memory.input.<file>`, so without this the file is loaded but unused —
    # the "I attached a file but nothing happened" case. Each active file is
    # embedded as a `<file>` / `<image>` block on top of the task.
    if files_edited and reads:
        blocks = [
            reads[cf.path].as_block(os.path.basename(cf.path) or cf.path)
            for cf in draft.active_files
            if cf.path and cf.path in reads
        ]
        joined = "\n\n".join(b for b in blocks if b)
        base_task = draft.task_description
        out["outer_context"] = (
            f"{base_task}\n\n{joined}" if joined else base_task
        )
    elif draft.task_edited:
        out["outer_context"] = draft.task_description

    # --- files=: also expose each file under its full path AND basename so a
    # chain that DOES reference `$memory.input.<basename>` resolves it.
    if files_edited:
        files_payload: dict[str, str] = dict(draft.extra_files)
        for cf in draft.active_files:
            if not cf.path or cf.path not in reads:
                continue
            value = reads[cf.path].memory_value
            files_payload.setdefault(cf.path, value)
            base = os.path.basename(cf.path)
            if not base or base in draft.extra_files:
                continue
            # A user-edited file (added / replaced) WINS the basename slot
            # over a saved file of the same name, so a freshly attached
            # replacement actually takes effect.
            if cf.status in ("added", "replaced"):
                files_payload[base] = value
            else:
                files_payload.setdefault(base, value)
        out["files"] = files_payload
        out["load_files_from_metadata"] = False
    return out


def _read_context_file(
    path: str, *, max_chars: int = MAX_CONTEXT_FILE_CHARS,
) -> tuple[str, bool]:
    """Back-compat thin wrapper over :func:`care.runtime.file_loading.load_file_text`."""
    return load_file_text(path, max_chars=max_chars)


def _read_file_safely(path: str) -> str:
    """Best-effort capped text read (kept for back-compat callers)."""
    return _read_context_file(path)[0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read(obj: Any, name: str) -> Any:
    """Read ``name`` off a model OR a dict."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


__all__ = [
    "ContextFile",
    "FileStatus",
    "RunContextDraft",
    "RunContextIssue",
    "add_file",
    "apply_overrides",
    "attach_path",
    "build_extra_kwargs",
    "compute_file_stat",
    "drop_file",
    "extract_run_context_draft",
    "missing_active_files",
    "replace_file",
    "resolve_file_arg",
    "restore_file",
    "set_model_override",
    "set_task",
    "validate_run_context_draft",
]
