"""
Chinese (zh_CN) Sphinx configuration.

Imports all settings from the parent English conf.py
and overrides language-specific settings.
"""

import os
import sys

# Add parent docs directory to path so we can import the main conf
sys.path.insert(0, os.path.abspath(".."))

# Import all settings from the English configuration
from conf import *  # noqa: F401, F403

# Override language settings
language = "zh_CN"
html_title = "SGLang 中文文档"
html_search_language = "zh"

# Adjust paths relative to this directory
templates_path = [os.path.join("..", "_templates")]
html_static_path = [os.path.join("..", "_static")]

# Fix logo/favicon paths relative to zh_CN directory
html_logo = os.path.join("..", "_static", "image", "logo.png")
html_favicon = os.path.join("..", "_static", "image", "logo.ico")

# Keep the same theme options but update for Chinese
html_theme_options.update(
    {
        "announcement": (
            '🌐 <a href="../en/index.html">English</a> | '
            "<strong>中文</strong>"
        ),
    }
)

html_context.update(
    {
        "current_language": "zh_CN",
        "languages": [
            ("en", "../en/index.html"),
            ("zh_CN", "../zh_CN/index.html"),
        ],
    }
)

# Source suffix - same as parent
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Exclude patterns
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
