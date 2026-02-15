import pytest

from local_rag_backend.prompting import (
    PromptTemplateError,
    render_prompt_template,
    validate_prompt_template,
)


def test_render_prompt_template_happy_path_and_escapes():
    tpl = "A {question} B {context} C {{literal}}"
    out = render_prompt_template(tpl, context="CTX", question="Q")
    assert out == "A Q B CTX C {literal}"


@pytest.mark.parametrize(
    "tpl",
    [
        "{question:1000000}",  # width spec DoS attempt in str.format
        "{question.__class__}",  # attribute access
        "{foo}",  # unknown placeholder
        "{",  # unmatched
        "}",  # unmatched
        "x {question",  # unmatched
        "x } y",  # unmatched
    ],
)
def test_prompt_template_rejects_unsafe_or_invalid_syntax(tpl: str):
    with pytest.raises(PromptTemplateError):
        validate_prompt_template(tpl)
