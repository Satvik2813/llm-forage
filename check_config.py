from forge.config import ModelConfig

c = ModelConfig(model="flash", temperature=0.7)
print("model:", c.model)
print("config:", c.to_gemini_config())

c2 = ModelConfig(model="flash", system="You are a tutor")
print("with system:", c2.to_gemini_config())

try:
    ModelConfig(model="flash", temperature=5.0)
    print("BUG: validation work avvatledu")
except ValueError as e:
    print("validation OK:", e)

from forge.client import LLMClient

llm = LLMClient(ModelConfig(model="flash"))

print(llm.chat("Say hi in exactly 3 words"))

for c in llm.stream("Count from 1 to 10 slowly"):
    print(c, end="", flush=True)

print()
print("Full response:", llm.last_response)