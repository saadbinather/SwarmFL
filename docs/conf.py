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
    "sphinx_copybutton",
    "sphinx_design",
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
# and repository files outside the Sphinx source tree. Experiment reports are
# linked from the Research index but omitted from toctrees to keep navigation
# compact, so their expected not-included notices are suppressed as well.
suppress_warnings = ["myst.xref_missing", "toc.not_included"]
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

html_theme = "pydata_sphinx_theme"
html_title = "SwarmFL documentation"
html_logo = "../resources/swarmfl-logo-transparent.png"
templates_path = ["_templates"]
html_sidebars = {"**": ["swarmfl-collapse.html", "swarmfl-nav.html"]}
html_static_path = ["_static"]
html_css_files = ["swarmfl.css"]

html_context = {
    "default_mode": "dark",
    "github_user": "saadbinather",
    "github_repo": "SwarmFL",
    "github_version": "main",
    "doc_path": "docs",
}
html_theme_options = {
    "logo": {
        "text": "",
    },
    "navbar_start": ["navbar-logo"],
    "navbar_center": [],
    "navbar_end": ["theme-switcher"],
    "navbar_persistent": ["search-button"],
    "show_nav_level": 1,
    "navigation_depth": 3,
    "collapse_navigation": True,
    "use_edit_page_button": True,
    "secondary_sidebar_items": ["page-toc"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/saadbinather/SwarmFL",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/swarmfl/",
            "icon": "fa-solid fa-box",
            "type": "fontawesome",
        },
    ],
}

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
