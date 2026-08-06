import warnings


# USD per 1 million tokens
PRICING = {
    "gemini-3.5-flash": {
        "input": 1.50,
        "output": 9.00,
    },
}


class UsageTracker:
    def __init__(self):
        self.call_count = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost = 0.0

        self.by_model = {}
        self._warned_models = set()

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ):
        input_tokens = input_tokens or 0
        output_tokens = output_tokens or 0

        self.call_count += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

        pricing = PRICING.get(model)

        if pricing is None:
            cost = 0.0

            if model not in self._warned_models:
                warnings.warn(
                    f"No pricing found for model '{model}'. "
                    "Cost will be recorded as $0.00.",
                    RuntimeWarning,
                    stacklevel=2,
                )

                self._warned_models.add(model)

        else:
            input_cost = (
                input_tokens / 1_000_000
            ) * pricing["input"]

            output_cost = (
                output_tokens / 1_000_000
            ) * pricing["output"]

            cost = input_cost + output_cost

        self.cost += cost

        if model not in self.by_model:
            self.by_model[model] = {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost": 0.0,
            }

        stats = self.by_model[model]

        stats["calls"] += 1
        stats["input_tokens"] += input_tokens
        stats["output_tokens"] += output_tokens
        stats["cost"] += cost

    def total_cost(self) -> float:
        return self.cost

    def summary(self) -> str:
        lines = [
            "Usage Summary",
            "-------------",
            f"Calls: {self.call_count}",
            f"Input tokens: {self.input_tokens:,}",
            f"Output tokens: {self.output_tokens:,}",
            f"Total tokens: "
            f"{self.input_tokens + self.output_tokens:,}",
            f"Estimated cost: ${self.cost:.6f}",
        ]

        if self.by_model:
            lines.append("")
            lines.append("By model:")

            for model, stats in self.by_model.items():
                lines.append(
                    f"  {model}: "
                    f"{stats['calls']} calls, "
                    f"{stats['input_tokens']:,} input, "
                    f"{stats['output_tokens']:,} output, "
                    f"${stats['cost']:.6f}"
                )

        return "\n".join(lines)