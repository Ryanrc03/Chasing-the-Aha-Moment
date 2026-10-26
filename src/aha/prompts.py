"""Prompt and output-format constants shared by training and evaluation."""

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
""".strip()

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""


def build_chat(question: str, system_prompt: str = SYSTEM_PROMPT) -> list[dict]:
    """Chat-format prompt: system instruction followed by the problem."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
