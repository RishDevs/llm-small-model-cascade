from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from cascade.prompts import label_schema, messages, parse_label


class ChatLabeler:
    def __init__(self, base_url, model, labels, constrained=True, timeout=30, workers=4, extra_body=None):
        self.client = OpenAI(base_url=base_url, api_key="unused", timeout=timeout, max_retries=1)
        self.model = model
        self.labels = list(labels)
        self.labelset = set(self.labels)
        self.constrained = constrained
        self.workers = workers
        self.extra_body = extra_body or {}

    def raw(self, text):
        extra = {}
        if self.constrained:
            extra["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "intent", "schema": label_schema(self.labels), "strict": True},
            }
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages(text, self.labels),
                temperature=0,
                max_tokens=32,
                extra_body=self.extra_body or None,
                **extra,
            )
        except Exception:
            return None
        return resp.choices[0].message.content or ""

    def label(self, text):
        return parse_label(self.raw(text), self.labelset)

    def __call__(self, texts):
        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(self.raw, texts))
