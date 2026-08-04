from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class ModelConfig:
    ALIASES = {
        "flash": "gemini-2.0-flash",
        "flash-lite": "gemini-2.0-flash-lite",
    }

    model: str
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 1024
    system: Optional[str] = None

    def __post_init__(self):
        self.model = self.ALIASES.get(self.model, self.model)

        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")

        if not 0 <= self.top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")

        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0")

    def to_gemini_config(self) -> dict:
        config = asdict(self)

        config.pop("model")
        config["max_output_tokens"] = config.pop("max_tokens")

        system = config.pop("system")
        if system is not None:
            config["system_instruction"] = system

        return config