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
    # Research results are confidential and are not published on the website.
    # The source files stay in the repository; only the build excludes them.
    "research.md",
    "status.md",
    "feedback-learning.md",
    "limitations.md",
    "next-steps.md",
    "research/reports.rst",
    "experiments/**",
    "community/**",
]

myst_fence_as_directive = ["mermaid"]
# Diagrams size to their content; colours come from swarmfl.css so they follow
# the light/dark switch without re-rendering.
mermaid_height = "auto"

autodoc_mock_imports = ["flwr", "numpy", "pydantic", "torch"]
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "show-inheritance": True,
}

html_theme = "pydata_sphinx_theme"
html_title = "SwarmFL"
html_logo = "../resources/swarmfl-logo-transparent.png"
html_favicon = "../resources/swarmfl-icon.png"
templates_path = ["_templates"]
html_sidebars = {"**": ["swarmfl-collapse.html", "swarmfl-nav.html"]}
html_static_path = ["_static"]
html_css_files = ["swarmfl.css"]
html_js_files = ["swarmfl-sidebar.js"]

html_context = {
    "default_mode": "dark",
    # Used by _templates/layout.html for link-preview (Open Graph) metadata.
    "swarmfl_site_url": "https://swarmfl.readthedocs.io/en/latest/",
    "swarmfl_description": (
        "SwarmFL is an experimental Python framework for federated multi-agent "
        "learning: independently managed specialists collaborate through Flower."
    ),
    "github_user": "saadbinather",
    "github_repo": "SwarmFL",
    "github_version": "main",
    "doc_path": "docs",
}
html_theme_options = {
    "logo": {
        "text": "",
    },
    "navbar_start": ["components/navbar-logo"],
    "navbar_center": [],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "show_nav_level": 1,
    "navigation_depth": 3,
    "collapse_navigation": True,
    "use_edit_page_button": False,
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
