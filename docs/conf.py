"""Sphinx configuration for the SwarmFL documentation site."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

project = "SwarmFL"
author = "SwarmFL contributors"
release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.mermaid",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
root_doc = "index"
# Historical reports intentionally retain links to local-only, ignored devlogs
# and repository files outside the Sphinx source tree.
suppress_warnings = ["myst.xref_missing"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "devlog/**",
]

myst_fence_as_directive = ["mermaid"]

autodoc_mock_imports = ["flwr", "numpy", "pydantic", "torch"]
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "show-inheritance": True,
}

html_theme = "sphinx_rtd_theme"
html_title = "SwarmFL 0.1.0 documentation"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
}
