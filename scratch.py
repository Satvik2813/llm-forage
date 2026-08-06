from forge.client import LLMClient
from forge.config import ModelConfig


config = ModelConfig(
    model="gemini-3.5-flash",
)

llm = LLMClient(config)


print("=== CHAT 1 ===")
print(llm.chat("hi"))

print("\n=== CHAT 2 ===")
print(llm.chat("hello there"))

print("\n=== STREAM ===")

for chunk in llm.stream("count to 3"):
    print(chunk, end="", flush=True)

print()


print("\n=== USAGE ===")
print(llm.usage.summary())