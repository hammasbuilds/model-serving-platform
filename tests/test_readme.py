"""Every ```python block in README.md must run verbatim, in a fresh namespace.

A README usage block once shadowed a version it never registered and raised
RegistryError on the first call. This test keeps that from coming back.
"""

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parent.parent / "README.md"
BLOCKS = re.findall(r"^```python\n(.*?)^```", README.read_text(encoding="utf-8"), re.M | re.S)


def test_readme_has_python_blocks():
    assert len(BLOCKS) >= 3


@pytest.mark.parametrize("code", BLOCKS, ids=[f"block{i}" for i in range(len(BLOCKS))])
def test_readme_python_block_runs_verbatim(code, capsys):
    exec(compile(code, str(README), "exec"), {"__name__": "__readme__"})
