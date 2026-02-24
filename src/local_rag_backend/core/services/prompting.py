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
    out: list[str] = []
    i = 0
    n = len(template)

    while i < n:
        ch = template[i]
        if ch == "{":
            if i + 1 < n and template[i + 1] == "{":
                out.append("{")
                i += 2
                continue
            j = template.find("}", i + 1)
            if j == -1:
                raise PromptTemplateError("Unmatched '{' in prompt_template.")
            key = template[i + 1 : j]
            if key == "context":
                out.append(context)
            elif key == "question":
                out.append(question)
            else:
                raise PromptTemplateError(
                    "Unsupported placeholder in prompt_template. "
                    "Only {context} and {question} are allowed (use {{ and }} for literals)."
                )
            i = j + 1
            continue

        if ch == "}":
            if i + 1 < n and template[i + 1] == "}":
                out.append("}")
                i += 2
                continue
            raise PromptTemplateError("Unmatched '}' in prompt_template.")

        out.append(ch)
        i += 1

    return "".join(out)


def validate_prompt_template(template: str) -> None:
    """Validate template syntax without producing large output."""
    # Rendering with empty strings is safe because we do not support width specs or
    # other format modifiers; output size is bounded by `len(template)`.
    render_prompt_template(template, context="", question="")
