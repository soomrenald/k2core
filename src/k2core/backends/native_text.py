"""Standalone Krea2 prompt template and tokenizer boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from k2core.inference.errors import ConfigurationError
from k2core.model import TokenizerReference, sha256_directory


KREA2_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
IM_START_TOKEN = 151644
USER_TOKEN = 872
NEWLINE_TOKEN = 198


@dataclass(frozen=True, slots=True)
class KreaPromptTokens:
    input_ids: tuple[int, ...]
    output_start: int

    @property
    def conditioned_ids(self) -> tuple[int, ...]:
        return self.input_ids[self.output_start :]


def load_tokenizer(reference: TokenizerReference):
    path = reference.path.expanduser().resolve(strict=True)
    observed = sha256_directory(path)
    if observed != reference.sha256:
        raise ConfigurationError(
            "Tokenizer assets do not match the registered identity.",
            technical_detail=f"expected {reference.sha256}, got {observed}",
            backend_name="native",
            phase="text_encoding",
        )
    try:
        from transformers import Qwen2Tokenizer
    except ImportError as error:
        raise ConfigurationError(
            "Native Krea2 prompt encoding requires Transformers.",
            technical_detail=str(error),
            backend_name="native",
            phase="text_encoding",
            remediation="Install K2Lab's model dependencies in the worker environment.",
        ) from error
    return Qwen2Tokenizer.from_pretrained(
        str(path),
        local_files_only=True,
    )


def tokenize_prompt(prompt: str, tokenizer: Any) -> KreaPromptTokens:
    encoded = tuple(
        int(token)
        for token in tokenizer.encode(
            KREA2_TEMPLATE.format(prompt),
            add_special_tokens=False,
        )
    )
    output_start = _conditioned_output_start(encoded)
    return KreaPromptTokens(input_ids=encoded, output_start=output_start)


def _conditioned_output_start(tokens: tuple[int, ...]) -> int:
    seen = 0
    template_end = -1
    for index, token in enumerate(tokens):
        if token == IM_START_TOKEN and seen < 2:
            template_end = index
            seen += 1
    if seen < 2:
        raise ValueError("Krea2 prompt template is missing its second <|im_start|> token")
    if len(tokens) > template_end + 2:
        if tokens[template_end + 1] == USER_TOKEN and tokens[template_end + 2] == NEWLINE_TOKEN:
            template_end += 3
    return template_end


__all__ = [
    "KREA2_TEMPLATE",
    "KreaPromptTokens",
    "load_tokenizer",
    "tokenize_prompt",
]
