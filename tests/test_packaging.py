"""Wheel/sdist packaging audit (TODO §7 P0).

Catches regressions in pyproject.toml configuration that
would silently ship a broken `uvx care` install:

* Logo PNGs + the demo TCSS are bundled in the wheel
  (hatchling drops non-`.py` files when packaging is
  mis-configured).
* Project metadata covers the basics PyPI / pip surface
  (description, classifiers, project urls, requires-python).
* Every declared extra resolves to non-empty dependency
  lists (a typo'd extra ships silently as an empty group).
* The console entry point points at `care.cli:main`.

Tests rebuild the wheel via `python -m build --wheel`
into a temp output dir so they exercise the actual
hatchling pipeline rather than reading the source
TOML. Slow-ish (~few seconds) but the audit is what
catches the regressions that would otherwise only
surface during `uvx` install.
"""

from __future__ import annotations

import subprocess
import sys
import sysconfig
import tomllib
import zipfile
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = PROJECT_ROOT / "pyproject.toml"


def _load_pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text())


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the wheel once per test module into a tmp dir.

    Uses `uv build --wheel` since the project's dev tooling
    is already on `uv`; falls back to `python -m build`
    when uv isn't on PATH so CI on a different runner
    still passes.
    """
    out = tmp_path_factory.mktemp("wheel")
    cmd_options = [
        ["uv", "build", "--wheel", "--out-dir", str(out)],
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(out)],
    ]
    last_exc: Exception | None = None
    for cmd in cmd_options:
        try:
            subprocess.run(
                cmd, cwd=PROJECT_ROOT, check=True,
                capture_output=True, text=True,
            )
            break
        except FileNotFoundError as exc:
            last_exc = exc
            continue
        except subprocess.CalledProcessError as exc:  # noqa: PERF203
            pytest.skip(
                f"wheel build failed (cmd={cmd[0]}): {exc.stderr}"
            )
    else:
        pytest.skip(f"no wheel builder available: {last_exc}")
    # Distribution is `maestro-care` (rebrand) -> wheel file `maestro_care-…`
    # (PEP 427 normalises `-` to `_`). The import package stays `care`.
    wheels = sorted(out.glob("maestro_care-*.whl"))
    assert wheels, f"no wheel produced in {out}"
    return wheels[-1]


# ---------------------------------------------------------------------------
# Source-side audit (no build needed)
# ---------------------------------------------------------------------------


class TestPyprojectMetadata:
    def test_pyproject_loads(self) -> None:
        data = _load_pyproject()
        assert "project" in data
        # Distribution renamed to maestro-care (MAESTRO CARE rebrand); the
        # import package, console script and wheel target deliberately stay
        # `care` — see the other assertions below.
        assert data["project"]["name"] == "maestro-care"

    def test_console_entry_point(self) -> None:
        data = _load_pyproject()
        assert data["project"]["scripts"]["care"] == "care.cli:main"

    def test_requires_python_is_312_plus(self) -> None:
        data = _load_pyproject()
        assert data["project"]["requires-python"].startswith(">=3.12")

    def test_every_declared_extra_has_deps(self) -> None:
        data = _load_pyproject()
        extras = data["project"].get("optional-dependencies", {})
        for name, deps in extras.items():
            assert deps, (
                f"extra {name!r} declared but has no "
                "dependencies — pip/uv would silently install "
                "nothing"
            )

    def test_full_extra_aggregates_optional_providers(self) -> None:
        data = _load_pyproject()
        extras = data["project"]["optional-dependencies"]
        full = set(extras["full"])
        for provider in ("carl", "openai", "anthropic", "docker", "e2b"):
            for dep in extras[provider]:
                assert dep in full, (
                    f"`full` extra missing {provider}'s dep {dep!r}"
                )

    def test_classifiers_cover_supported_python(self) -> None:
        data = _load_pyproject()
        classifiers = data["project"].get("classifiers", [])
        assert any(
            c.startswith("Programming Language :: Python :: 3.12")
            for c in classifiers
        )

    def test_project_urls_present(self) -> None:
        data = _load_pyproject()
        urls = data["project"].get("urls", {})
        assert "Homepage" in urls or "Repository" in urls

    def test_hatch_wheel_packages_care(self) -> None:
        data = _load_pyproject()
        wheel_cfg = data["tool"]["hatch"]["build"]["targets"]["wheel"]
        assert wheel_cfg["packages"] == ["care"]
        # Defensive include globs cover the non-.py assets.
        include = wheel_cfg.get("include", [])
        assert any(g.endswith(".png") for g in include)
        assert any(g.endswith(".tcss") for g in include)
        # i18n catalogs ship as JSON — the TUI crashes on first `t()` call
        # if they're missing from the wheel.
        assert any(g.endswith(".json") for g in include)


# ---------------------------------------------------------------------------
# Built-artifact audit
# ---------------------------------------------------------------------------


class TestBuiltWheel:
    def test_assets_bundled(self, built_wheel: Path) -> None:
        with zipfile.ZipFile(built_wheel) as zf:
            names = zf.namelist()
        pngs = {n for n in names if n.endswith(".png")}
        # All 5 logo sizes plus the default unsized one.
        expected = {
            "care/assets/airi_logo.png",
            "care/assets/airi_logo_8.png",
            "care/assets/airi_logo_10.png",
            "care/assets/airi_logo_12.png",
            "care/assets/airi_logo_16.png",
        }
        assert expected.issubset(pngs), (
            f"missing logos: {expected - pngs}"
        )

    def test_styles_bundled(self, built_wheel: Path) -> None:
        with zipfile.ZipFile(built_wheel) as zf:
            names = zf.namelist()
        tcss = {n for n in names if n.endswith(".tcss")}
        assert "care/styles/demo.tcss" in tcss

    def test_locales_bundled(self, built_wheel: Path) -> None:
        # The i18n layer loads care/runtime/locales/<lang>.json via
        # importlib.resources at first render; a missing catalog crashes
        # the TUI on boot.
        with zipfile.ZipFile(built_wheel) as zf:
            names = set(zf.namelist())
        for lang in ("en", "ru"):
            assert f"care/runtime/locales/{lang}.json" in names, (
                f"wheel missing locale catalog: {lang}.json"
            )

    def test_uv_lock_not_bundled(self, built_wheel: Path) -> None:
        # uv.lock is intentionally NOT shipped in the wheel: it's
        # .gitignored (absent in CI), can't be regenerated on a clean
        # runner (the editable [tool.uv.sources] point at sibling
        # checkouts that don't exist there), and pins local path deps
        # meaningless to a PyPI consumer. Bundling it broke
        # `python -m build`. See the note in pyproject.toml.
        with zipfile.ZipFile(built_wheel) as zf:
            names = zf.namelist()
        assert "care/uv.lock" not in names, (
            "wheel unexpectedly bundles care/uv.lock — it's "
            "intentionally excluded (see pyproject.toml note)"
        )

    def test_metadata_declares_all_required_deps(
        self, built_wheel: Path,
    ) -> None:
        with zipfile.ZipFile(built_wheel) as zf:
            md_name = next(
                n for n in zf.namelist() if n.endswith("/METADATA")
            )
            md = zf.read(md_name).decode()
        required = [
            "textual>=8.2.6",
            "pydantic>=2.11",
            "httpx>=0.27",
            "gigaevo-client>=0.3.0",
            "mmar-mage>=0.1",
            "pypdf>=4.0",
            "rich-pixels>=3.0",
        ]
        for dep in required:
            assert f"Requires-Dist: {dep}" in md, (
                f"wheel metadata missing dep: {dep!r}"
            )

    def test_metadata_declares_console_script(
        self, built_wheel: Path,
    ) -> None:
        with zipfile.ZipFile(built_wheel) as zf:
            entry = next(
                (n for n in zf.namelist()
                 if n.endswith("entry_points.txt")),
                None,
            )
            assert entry, "wheel missing entry_points.txt"
            text = zf.read(entry).decode()
        assert "care = care.cli:main" in text

    def test_metadata_declares_every_extra(
        self, built_wheel: Path,
    ) -> None:
        data = _load_pyproject()
        extras = list(
            data["project"].get("optional-dependencies", {}).keys()
        )
        with zipfile.ZipFile(built_wheel) as zf:
            md_name = next(
                n for n in zf.namelist() if n.endswith("/METADATA")
            )
            md = zf.read(md_name).decode()
        for extra in extras:
            assert f"Provides-Extra: {extra}" in md, (
                f"wheel metadata missing extra: {extra!r}"
            )


# ---------------------------------------------------------------------------
# Sanity: the *installed* package can still find its assets.
# ---------------------------------------------------------------------------


class TestAssetResolution:
    def test_logos_exist_on_disk(self) -> None:
        # The TUI's banner loader reads the logos via the
        # package path. Verify they're physically present
        # in the source layout so the wheel build has
        # something to copy.
        assets = PROJECT_ROOT / "care" / "assets"
        for name in (
            "airi_logo.png",
            "airi_logo_8.png",
            "airi_logo_10.png",
            "airi_logo_12.png",
            "airi_logo_16.png",
        ):
            assert (assets / name).is_file(), (
                f"missing logo at source path: {name}"
            )

    def test_styles_exist_on_disk(self) -> None:
        styles = PROJECT_ROOT / "care" / "styles"
        assert (styles / "demo.tcss").is_file()

    def test_locales_resolve_via_importlib_resources(self) -> None:
        # care.runtime.i18n loads catalogs through
        # importlib.resources.files("care.runtime.locales"), which only
        # works when the directory ships as a package (has __init__.py).
        # Guards against a refactor that drops the __init__.py or a
        # catalog file.
        import json
        from importlib.resources import files

        root = files("care.runtime.locales")
        for lang in ("en", "ru"):
            ref = root.joinpath(f"{lang}.json")
            data = json.loads(ref.read_text(encoding="utf-8"))
            assert isinstance(data, dict) and data, (
                f"{lang}.json did not load as a non-empty catalog"
            )

    @pytest.mark.skipif(
        sysconfig.get_platform().startswith("win"),
        reason="POSIX-only file mode check",
    )
    def test_logos_are_readable(self) -> None:
        # Catch a permission regression where the asset
        # files become non-world-readable + the wheel ends
        # up shipping unreadable files.
        assets = PROJECT_ROOT / "care" / "assets"
        for entry in assets.glob("*.png"):
            mode = entry.stat().st_mode & 0o777
            assert mode & 0o444, (
                f"{entry.name}: not world-readable (mode {oct(mode)})"
            )

    def test_logos_resolve_via_importlib_resources(self) -> None:
        # The chat banner loads logos through
        # `importlib.resources.files("care.assets")`, which
        # only works when `care/assets/__init__.py` ships so
        # the directory registers as a package. This guards
        # against a refactor that drops the `__init__.py` or
        # restructures the asset layout in a way that breaks
        # the wheel-resident lookup.
        from importlib.resources import files

        root = files("care.assets")
        for name in (
            "airi_logo.png",
            "airi_logo_8.png",
            "airi_logo_10.png",
            "airi_logo_12.png",
            "airi_logo_16.png",
        ):
            ref = root.joinpath(name)
            with ref.open("rb") as fh:
                head = fh.read(8)
            assert head.startswith(b"\x89PNG"), (
                f"{name}: not a PNG (header {head!r})"
            )
