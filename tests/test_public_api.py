"""The README's quickstart imports straight from the top-level package - that has
to actually work, and pip's fallback install path has to actually install pytest.

Both were silently broken: `src/serving/__init__.py` re-exported nothing, so
`from serving import ServingPlatform` (the first line of code every reader of the
README pastes) raised ImportError; and `pip install -e ".[dev]"` (the documented
fallback for anyone without `uv`) installed zero dev dependencies because `dev` was
only declared as a PEP 735 dependency group, not a real extra.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_readme_quickstart_import_works():
    """Exactly the line from README's 'Run it yourself' and 'Usage' sections."""
    from serving import ServingPlatform

    platform = ServingPlatform()
    platform.registry.register("m", 1)
    platform.load("m", 1, lambda features: "ok")
    platform.registry.promote("m", 1)
    assert platform.predict("m", {}).output == "ok"


def test_public_api_exports_the_names_a_caller_needs():
    import serving

    for name in (
        "ServingPlatform",
        "ServingError",
        "Registry",
        "RegistryError",
        "SLO",
        "SLOMonitor",
        "Prediction",
        "Stage",
        "ModelVersion",
    ):
        assert hasattr(serving, name), f"serving.{name} is not exported"


def test_pip_dev_extra_is_a_real_extra_not_only_a_dependency_group():
    """`pip install -e ".[dev]"` (the README's no-uv fallback) must actually
    install something. A pip that only understands PEP 621 extras - which is any
    plain pip, with no PEP 735 support - needs `dev` under
    [project.optional-dependencies], not only under [dependency-groups]."""
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    extras = data["project"]["optional-dependencies"]
    assert "dev" in extras
    assert any("pytest" in dep for dep in extras["dev"])
