import json
import re

SYSTEM = """You label messages sent to a virtual assistant with their intent.

Pick exactly one label from this list:
{labels}

Use "oos" when the message does not match any label on the list.
Reply with the label only."""


def system_prompt(labels):
    return SYSTEM.format(labels="\n".join(labels))


def messages(text, labels):
    return [
        {"role": "system", "content": system_prompt(labels)},
        {"role": "user", "content": text},
    ]


def label_schema(labels):
    return {
        "type": "object",
        "properties": {"label": {"type": "string", "enum": list(labels)}},
        "required": ["label"],
        "additionalProperties": False,
    }


def parse_label(raw, labelset):
    if raw is None:
        return None
    text = raw.strip()
    if text.startswith("{"):
        try:
            text = str(json.loads(text).get("label", ""))
        except (json.JSONDecodeError, AttributeError):
            return None
    text = text.strip().strip("\"'`.").lower()
    text = re.sub(r"[\s\-]+", "_", text)
    return text if text in labelset else None
