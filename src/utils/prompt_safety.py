"""Helpers for separating untrusted text from LLM instructions."""


def untrusted_text(label: str, value: str, max_length: int | None = None) -> str:
    """Wrap external text as data and normalize control characters before prompting."""
    text = str(value).replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    if max_length is not None:
        text = text[:max_length]
    return f"--- BEGIN UNTRUSTED {label} ---\n{text}\n--- END UNTRUSTED {label} ---"
