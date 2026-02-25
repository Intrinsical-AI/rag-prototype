"""
Prompt template rendering with strict safety constraints.

Why:
- `str.format()` is powerful enough to be abused for memory DoS via gigantic width specs
  (e.g. "{question:100000000}") and for unexpected attribute/index access.
- This project allows a per-request `prompt_template` in `/ask_eval`, so the template must
  be treated as untrusted input.

This renderer only supports the literal placeholders `{context}` and `{question}` plus
escaped braces `{{` and `}}`.
"""

from __future__ import annotations


class PromptTemplateError(ValueError):
    """Raised when a prompt template contains unsupported / unsafe syntax."""


def render_prompt_template(template: str, *, context: str, question: str) -> str:
    """
    Render a prompt template safely.

    Supported:
    - `{context}` -> replaced with `context`
    - `{question}` -> replaced with `question`
    - `{{` and `}}` -> literal braces
    """
    to_render: list[str] = []
    left_idx = 0
    len_ = len(template)

    while left_idx < len_:
        current_ch = template[left_idx]
        if current_ch == "{":
            if left_idx + 1 < len_ and template[left_idx + 1] == "{":
                to_render.append("{")
                left_idx += 2
                continue
            right_idx = template.find("}", left_idx + 1)
            if right_idx == -1:
                raise PromptTemplateError("Unmatched '{' in prompt_template.")
            key = template[left_idx + 1 : right_idx]
            if key == "context":
                to_render.append(context)
            elif key == "question":
                to_render.append(question)
            else:
                raise PromptTemplateError(
                    "Unsupported placeholder in prompt_template. "
                    "Only {context} and {question} are allowed (use {{ and }} for literals)."
                )
            left_idx = right_idx + 1
            continue

        if current_ch == "}":
            if left_idx + 1 < len_ and template[left_idx + 1] == "}":
                to_render.append("}")
                left_idx += 2
                continue
            raise PromptTemplateError("Unmatched '}' in prompt_template.")

        to_render.append(current_ch)
        left_idx += 1

    return "".join(to_render)


def validate_prompt_template(template: str) -> None:
    """Validate template syntax without producing large output."""
    # Rendering with empty strings is safe because we do not support width specs or
    # other format modifiers; output size is bounded by `len(template)`.
    render_prompt_template(template, context="", question="")
