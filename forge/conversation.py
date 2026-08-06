import json


class Conversation:
    def __init__(self):
        self.messages = []

    def add_user(self, text: str):
        self.messages.append(
            {
                "role": "user",
                "parts": [{"text": text}],
            }
        )

    def add_model(self, text: str):
        self.messages.append(
            {
                "role": "model",
                "parts": [{"text": text}],
            }
        )

    def remove_last(self):
        if self.messages:
            self.messages.pop()

    def to_contents(self):
        return self.messages

    def clear(self):
        self.messages.clear()

    def trim(self, max_messages: int):
        if max_messages < 0:
            raise ValueError("max_messages must be >= 0")

        max_messages -= max_messages % 2

        if max_messages == 0:
            self.clear()
            return

        if len(self.messages) <= max_messages:
            return

        drop_count = len(self.messages) - max_messages

        if drop_count % 2 != 0:
            drop_count += 1

        self.messages = self.messages[drop_count:]

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as file:
            json.dump(
                self.messages,
                file,
                indent=4,
                ensure_ascii=False,
            )

    @classmethod
    def load(cls, path: str):
        conversation = cls()

        with open(path, "r", encoding="utf-8") as file:
            conversation.messages = json.load(file)

        return conversation

    def __len__(self):
        return len(self.messages)