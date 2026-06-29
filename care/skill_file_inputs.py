"""Detect file-consuming AgentSkill steps and wire an attached file into them.

When a generated chain's first step is a document skill (``docx`` / ``pdf`` /
``xlsx`` / ``pptx``) that *reads* a file — e.g. "extract text from the provided
DOCX" — MAGE authors it referencing the document via ``$outer_context`` (the
task text) or with an empty ``input_mapping``. Neither actually delivers a
file, so the chat surface runs the chain against prose instead of a document
(the "it ran without my file" surprise).

CARL/MAGE are upstream-frozen here, so the bridge lives entirely in CARE:

* :func:`requires_file_input` / :func:`file_consuming_steps` — detect the
  read-a-document steps so the chat can prompt the user to attach a file
  *before* running.
* :func:`apply_file_inputs` — rewrite each such step so it reads the document
  from ``memory["input"][<basename>]`` (a CARL-resolvable ``$memory.input.*``
  reference), and return the ``{basename: text}`` dict CARE feeds into
  ``build_run_context`` (which seeds ``context.memory["input"]``). CARE extracts
  the document to text with the same ``document_extract`` path the chat's
  ``@file`` refs already use, so this needs no sandbox / skill download.

The transform is deliberately conservative: it only touches steps whose skill
is a known document reader AND whose intent is to *read* (not *create*) a file,
and it leaves every other step untouched.
"""

from __future__ import annotations

import copy
import os
import re
from typing import Any, Callable, Optional

#: A read-vs-create classifier: ``(step) -> True`` (reads a file) / ``False``
#: (creates / no file) / ``None`` (undecided → fall back to the heuristic).
ReadsPredicate = Callable[[dict], Optional[bool]]

# Known Anthropic document skills that consume a file on input. Matched by the
# slug in the skill reference (``github://anthropics/skills/skills/docx@main``
# → ``docx``). ``pptx``/``xlsx``/``docx`` can also *create* files, so a slug
# match alone isn't enough — see :func:`_reads_a_file`.
FILE_SKILL_SLUGS: tuple[str, ...] = ("docx", "pdf", "xlsx", "pptx")

# Read-intent keywords (en + ru). A doc-skill step counts as "needs a file on
# input" only when its title / aim / task talk about reading the document.
# Deliberately excludes "summarise" — that's a downstream LLM step, not a
# file read, and would mis-flag a "create a doc with the summary" step.
_READ_INTENT = re.compile(
    r"extract|parse|\bread\b|convert from|provided|attached|"
    r"извлеч|прочит|разбор|из файла|из docx|из pdf|приложен|"
    r"переданн|содержим",
    re.IGNORECASE,
)
# Create-intent keywords — a "create a Word doc" step needs NO file input, so it
# must NOT trigger an attach prompt even though its skill slug is a doc skill.
_CREATE_INTENT = re.compile(
    r"\bcreate\b|\bgenerate\b|\bbuild\b|\bwrite a\b|\bproduce\b|\bauthor\b|"
    r"создать|сгенерир|сформир|построить|написать документ|подготовить файл",
    re.IGNORECASE,
)


def _step_config(step: Any) -> dict:
    cfg = step.get("step_config") if isinstance(step, dict) else None
    return cfg if isinstance(cfg, dict) else {}


def _skill_ref(cfg: dict) -> str:
    """The skill reference, as a string, in either form it can take.

    MAGE emits ``skill`` as a string URI
    (``github://anthropics/skills/skills/docx@main``); ``ReasoningChain
    .to_dict()`` serialises it as a structured ``SkillReference`` dict
    (``{"git_subdirectory": "skills/docx", "git_url": …}``). Handle both so
    detection works on the chat (raw dict) AND library/CLI (round-tripped)
    paths.
    """
    val = cfg.get("skill")
    if isinstance(val, str) and val:
        return val
    if isinstance(val, dict):
        for key in (
            "git_subdirectory", "name", "path",
            "package_subpath", "package",
        ):
            sub = val.get(key)
            if isinstance(sub, str) and sub:
                return sub
    for key in ("skill_uri", "skill_id", "skill_name"):
        sub = cfg.get(key)
        if isinstance(sub, str) and sub:
            return sub
    return ""


def skill_slug(ref: str) -> str:
    """Extract the bare skill slug from a skill reference.

    ``github://anthropics/skills/skills/docx@main`` → ``docx``;
    ``pdf-extractor`` → ``pdf-extractor``; ``""`` → ``""``.
    """
    if not ref:
        return ""
    tail = ref.split("/")[-1]
    tail = tail.split("@", 1)[0]
    return tail.strip().lower()


def _is_doc_skill(cfg: dict) -> bool:
    slug = skill_slug(_skill_ref(cfg))
    if not slug:
        return False
    return any(slug == s or slug.startswith(s + "-") for s in FILE_SKILL_SLUGS)


def _reads_a_file(step: dict, cfg: dict) -> bool:
    """Heuristic: does this doc-skill step READ a file (vs create one)?"""
    blob = " ".join(
        str(step.get(k) or "")
        for k in ("title", "aim", "stage_action")
    )
    blob += " " + str(cfg.get("task") or "")
    if _CREATE_INTENT.search(blob) and not _READ_INTENT.search(blob):
        return False
    # Default to True for a doc skill: reading is the common case and a missing
    # input is the failure we're guarding against. An explicit create-only verb
    # (handled above) is the sole opt-out.
    return True


def doc_skill_steps(chain_dict: Any) -> list[dict]:
    """Return every ``agent_skill`` step whose skill is a document skill
    (docx/pdf/xlsx/pptx) — the candidates for "reads a file" classification,
    BEFORE the read-vs-create decision."""
    if not isinstance(chain_dict, dict):
        return []
    return [
        step
        for step in chain_dict.get("steps") or []
        if isinstance(step, dict)
        and step.get("step_type") == "agent_skill"
        and _is_doc_skill(_step_config(step))
    ]


def file_consuming_steps(
    chain_dict: Any, *, reads: "ReadsPredicate | None" = None,
) -> list[dict]:
    """Return the ``agent_skill`` steps that READ a document on input.

    ``reads`` is an optional classifier ``(step) -> bool | None`` — when it
    returns ``None`` (or isn't supplied) the read-vs-create decision falls
    back to the keyword heuristic :func:`_reads_a_file`. The LLM-backed
    classifier (:func:`classify_reads_intent` + :func:`reads_predicate`) plugs
    in here so the model — not regex — decides, with the heuristic as a safety
    net.
    """
    out: list[dict] = []
    for step in doc_skill_steps(chain_dict):
        cfg = _step_config(step)
        verdict: Optional[bool] = None
        if reads is not None:
            try:
                verdict = reads(step)
            except Exception:  # noqa: BLE001 — classifier never breaks detection
                verdict = None
        if verdict is None:
            verdict = _reads_a_file(step, cfg)
        if verdict:
            out.append(step)
    return out


def requires_file_input(
    chain_dict: Any, *, reads: "ReadsPredicate | None" = None,
) -> bool:
    """``True`` when the chain has a document-reading skill step — i.e. it
    expects a file the caller must supply before the run is meaningful."""
    return bool(file_consuming_steps(chain_dict, reads=reads))


def _input_key(basename: str) -> str:
    """Sanitise a basename into a stable ``memory["input"]`` key.

    Keeps it human-readable (the chain step interpolates it) while staying a
    single dotted-path-safe token: spaces → ``_``; everything outside
    ``[A-Za-z0-9._-]`` dropped.
    """
    key = re.sub(r"\s+", "_", basename.strip())
    key = re.sub(r"[^A-Za-z0-9._-]", "", key)
    return key or "document"


def apply_file_inputs(
    chain_dict: dict,
    attachments: list[tuple[str, str]],
    *,
    reads: "ReadsPredicate | None" = None,
) -> tuple[dict, dict[str, str]]:
    """Wire attached document(s) into the chain's file-consuming skill steps.

    Args:
        chain_dict: The generated chain (mutated on a copy, not in place).
        attachments: ``[(path, extracted_text), …]`` — the file CARE already
            extracted to text via ``document_extract`` (same as a chat
            ``@file`` ref).

    Returns:
        ``(new_chain_dict, files)`` where ``files`` is the
        ``{input_key: text}`` dict to pass as ``build_run_context(files=…)``
        (it seeds ``context.memory["input"]``). Each file-consuming step's
        ``input_mapping`` is rewritten to read the (first) document from
        ``$memory.input.<input_key>`` so the step actually sees the content
        regardless of how MAGE originally (mis)wired it.

    No-op (returns the chain unchanged + ``{}``) when there are no attachments
    or no file-consuming steps.
    """
    steps = file_consuming_steps(chain_dict, reads=reads)
    if not attachments or not steps:
        return chain_dict, {}
    target_numbers = {s.get("number") for s in steps}

    files: dict[str, str] = {}
    keys: list[str] = []
    for path, text in attachments:
        base = os.path.basename(path) or "document"
        key = _input_key(base)
        # De-dupe collided keys so two attachments don't clobber each other.
        n, uniq = 1, key
        while uniq in files:
            n += 1
            uniq = f"{key}-{n}"
        files[uniq] = text or ""
        keys.append(uniq)

    primary = keys[0]
    new_chain = copy.deepcopy(chain_dict)
    # deepcopy broke object identity; rewrite exactly the detected
    # file-consuming steps (matched by number) to read the primary document.
    for step in new_chain.get("steps") or []:
        if not isinstance(step, dict) or step.get("number") not in target_numbers:
            continue
        cfg = _step_config(step)
        param = _file_param_name(cfg)
        mapping = dict(cfg.get("input_mapping") or {})
        mapping[param] = f"$memory.input.{primary}"
        cfg["input_mapping"] = mapping
        cfg["task"] = _inject_document_placeholder(
            str(cfg.get("task") or ""), param, primary,
        )
        step["step_config"] = cfg

    return new_chain, files


def _inject_document_placeholder(task: str, param: str, label: str) -> str:
    """Append a ``{param}`` placeholder to a skill ``task`` so the resolved
    document text is actually interpolated into the LLM prompt.

    CARL's ``_interpolate_task`` does ``task.format_map(resolved_inputs)`` — the
    resolved input only reaches the model when the task string references it as
    ``{param}``. MAGE authors a task that *mentions* a file in prose but has no
    placeholder, so the document is silently dropped. We escape the original
    task's literal braces (so a stray ``{`` can't break ``format_map`` and a
    ``KeyError`` won't discard the whole substitution) and append a single real
    placeholder bound to the document. Idempotent: a task that already
    references ``{param}`` is returned unchanged.
    """
    placeholder = "{" + param + "}"
    if placeholder in task:
        return task
    safe = task.replace("{", "{{").replace("}", "}}")
    return (
        f"{safe}\n\n"
        f"--- Attached document ({label}) ---\n"
        f"{placeholder}"
    )


def _file_param_name(cfg: dict) -> str:
    """Pick the input_mapping key to bind the document to.

    Reuse an existing file-ish param if the step already declared one (so the
    skill's task interpolation still lines up); otherwise default to ``file``.
    """
    mapping = cfg.get("input_mapping")
    if isinstance(mapping, dict):
        for cand in ("file", "path", "file_path", "document", "doc", "input"):
            if cand in mapping:
                return cand
        # Fall back to the first existing key (single-input skills).
        for k in mapping:
            if isinstance(k, str) and k:
                return k
    return "file"


# ---------------------------------------------------------------------------
# Model-based read-vs-create classification (replaces the regex heuristic on
# the interactive paths; the heuristic stays as the offline / fallback path)
# ---------------------------------------------------------------------------


async def classify_reads_intent(
    api: Any, steps: list[dict],
) -> dict[int, bool]:
    """Ask the LLM which document-skill ``steps`` READ a user-provided file.

    Returns ``{step_number: reads_bool}``. Empty ``{}`` on any failure (no
    ``api``, network error, unparseable reply) so the caller transparently
    falls back to the keyword heuristic per step. One short call classifies
    all candidate steps at once.
    """
    if not steps or api is None:
        return {}
    lines = []
    for s in steps:
        cfg = _step_config(s)
        lines.append(
            f"- step {s.get('number')}: title={s.get('title')!r} "
            f"aim={s.get('aim')!r} task={(str(cfg.get('task') or ''))[:200]!r}"
        )
    prompt = (
        "You classify steps in an agent pipeline. For each step decide if it "
        "READS an input document/file that the USER must provide before the "
        "run (true), or if it CREATES/produces a file or doesn't need a "
        "user-provided file (false).\n\n"
        + "\n".join(lines)
        + "\n\nReturn ONLY a JSON object mapping each step number (as a "
        'string) to a boolean, e.g. {"1": true, "2": false}. No prose.'
    )
    try:
        raw = await api.get_response_with_retries(prompt, 2)
    except Exception:  # noqa: BLE001 — fall back to the heuristic
        return {}
    return _parse_verdicts(raw)


def _parse_verdicts(raw: Any) -> dict[int, bool]:
    import json

    if not isinstance(raw, str):
        return {}
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except (ValueError, TypeError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[int, bool] = {}
    for key, val in data.items():
        try:
            out[int(key)] = bool(val)
        except (ValueError, TypeError):
            continue
    return out


def reads_predicate(verdicts: dict[int, bool]) -> ReadsPredicate:
    """Wrap LLM ``verdicts`` (from :func:`classify_reads_intent`) into a
    :data:`ReadsPredicate`. Unclassified steps return ``None`` so detection
    falls back to the keyword heuristic for them."""

    def _reads(step: dict) -> Optional[bool]:
        return verdicts.get(step.get("number"))

    return _reads


async def classify_reads(
    api: Any, chain_dict: Any,
) -> "ReadsPredicate | None":
    """Convenience: classify a chain's document-skill steps and return a ready
    :data:`ReadsPredicate` (or ``None`` when there's nothing to classify /
    no usable verdict — caller then uses the heuristic)."""
    candidates = doc_skill_steps(chain_dict)
    if not candidates:
        return None
    verdicts = await classify_reads_intent(api, candidates)
    return reads_predicate(verdicts) if verdicts else None


__all__ = [
    "FILE_SKILL_SLUGS",
    "ReadsPredicate",
    "apply_file_inputs",
    "classify_reads",
    "classify_reads_intent",
    "doc_skill_steps",
    "file_consuming_steps",
    "reads_predicate",
    "requires_file_input",
    "skill_slug",
]
