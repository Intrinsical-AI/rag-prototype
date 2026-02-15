"""
Backward-compatible re-export of prompting utilities.

Deprecated: import from `local_rag_backend.core.services.prompting` instead.
"""

from __future__ import annotations

from local_rag_backend.core.services.prompting import (
    PromptTemplateError,
    render_prompt_template,
    validate_prompt_template,
)

__all__ = ["PromptTemplateError", "render_prompt_template", "validate_prompt_template"]
