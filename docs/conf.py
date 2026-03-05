# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime
import os
import sys

import mimiqcircuits
from mimiqcircuits.__version__ import __version__

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "MIMIQ Circuits"
copyright = f"2023-{datetime.date.today().year}, QPerfect"
author = "QPerfect"
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

autosummary_generate = True

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_inline_tabs",
    "sphinx.ext.doctest",
    "myst_parser",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".pytest_cache"]

# Tell Sphinx to look for markdown files

master_doc = "index"

# -- Napoleon settings -------------------------------------------------------
# For Google/NumPy docstring styles

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Autodoc settings --------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
}

autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# -- Copy button configuration -----------------------------------------------

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = True

# -- Intersphinx mapping -----------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# -- Options for LaTeX output ------------------------------------------------

latex_engine = "xelatex"
latex_elements = {
    "preamble": r"\usepackage{physics}",
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_title = f"{project} v{release}"
html_logo = "_static/qperfect_logo.png"
html_static_path = ["_static"]
html_css_files = [
    "custom.css",
]

# Pygments syntax highlighting themes
pygments_style = "friendly"
pygments_dark_style = "monokai"

# Furo theme options
html_theme_options = {
    # Announcement banner
    "announcement": f"The MIMIQ Autumn Update 2024 is out! Update now to the latest <strong>v{release}</strong> version.",
    # TOC depth
    "navigation_with_keys": True,
    # Light mode CSS variables
    "light_css_variables": {
        # Brand
        "color-brand-primary": "#990066",
        "color-brand-content": "#990066",
        # Inline code
        "color-inline-code-background": "rgba(153, 0, 102, 0.08)",
        # Admonitions
        "color-admonition-background": "#f8f9fc",
        # API
        "color-api-name": "#990066",
        "color-api-pre-name": "#4b5563",
        # Fonts
        "font-stack": "'Inter', -apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif, Apple Color Emoji, Segoe UI Emoji",
        "font-stack--headings": "'Rajdhani', -apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif, Apple Color Emoji, Segoe UI Emoji",
        "font-stack--monospace": "'JetBrains Mono', 'Courier New', Courier, monospace",
        "font-weight--headings": "600",
    },
    # Dark mode CSS variables
    "dark_css_variables": {
        # Brand
        "color-brand-primary": "#d94faa",
        "color-brand-content": "#d94faa",
        # Backgrounds
        "color-background-primary": "#0f1117",
        "color-background-secondary": "#161922",
        "color-background-border": "#2d3548",
        # Foreground
        "color-foreground-primary": "#f3f4f6",
        "color-foreground-secondary": "#d1d5db",
        "color-foreground-muted": "#9ca3af",
        # Sidebar
        "color-sidebar-background": "#0d0f14",
        "color-sidebar-background-border": "#1f2937",
        "color-sidebar-link-text": "#e5e7eb",
        "color-sidebar-link-text--top-level": "#ffffff",
        "color-sidebar-item-background--current": "rgba(217, 79, 170, 0.15)",
        "color-sidebar-item-background--hover": "rgba(217, 79, 170, 0.08)",
        # Inline code
        "color-inline-code-background": "rgba(249, 168, 212, 0.15)",
        # Admonitions
        "color-admonition-background": "#1a1f2c",
        # API
        "color-api-background": "#1a1f2c",
        "color-api-name": "#d94faa",
        "color-api-pre-name": "#9ca3af",
        # Highlighted text
        "color-highlight-on-target": "rgba(255, 215, 0, 0.15)",
        # Fonts
        "font-stack": "'Inter', -apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif, Apple Color Emoji, Segoe UI Emoji",
        "font-stack--headings": "'Rajdhani', -apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif, Apple Color Emoji, Segoe UI Emoji",
        "font-stack--monospace": "'JetBrains Mono', 'Courier New', Courier, monospace",
        "font-weight--headings": "600",
    },
    # Footer icons
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/qperfect-io/mimiqcircuits-python",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
        {
            "name": "QPerfect",
            "url": "https://qperfect.io",
            "html": """
                <svg stroke="currentColor" fill="none" stroke-width="1.5" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M12 21a9.004 9.004 0 008.716-6.747M12 21a9.004 9.004 0 01-8.716-6.747M12 21c2.485 0 4.5-4.03 4.5-9S14.485 3 12 3m0 18c-2.485 0-4.5-4.03-4.5-9S9.515 3 12 3m0 0a8.997 8.997 0 017.843 4.582M12 3a8.997 8.997 0 00-7.843 4.582m15.686 0A11.953 11.953 0 0112 10.5c-2.998 0-5.74-1.1-7.843-2.918m15.686 0A8.959 8.959 0 0121 12c0 .778-.099 1.533-.284 2.253m0 0A17.919 17.919 0 0112 16.5c-3.162 0-6.133-.815-8.716-2.247m0 0A9.015 9.015 0 013 12c0-1.605.42-3.113 1.157-4.418" />
                </svg>
            """,
            "class": "",
        },
    ],
    # Source repository link
    "source_repository": "https://github.com/qperfect-io/mimiqcircuits-python",
    "source_branch": "main",
    "source_directory": "docs/",
}
