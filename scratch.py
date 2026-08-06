from forge.config import ModelConfig
from forge.client import LLMClient
from forge.conversation import Conversation

llm = LLMClient(ModelConfig(model="flash"))

print("--- memory ---")
conv = Conversation()
llm.chat("My name is Satvik", conv)
print(llm.chat("What's my name?", conv))
print("len:", len(conv))

print("--- orphan chat ---")
bad = LLMClient(ModelConfig(model="does-not-exist"))
conv2 = Conversation()
try:
    bad.chat("hi", conv2)
except Exception:
    pass
print("len:", len(conv2))

print("--- orphan stream ---")
conv3 = Conversation()
try:
    for c in bad.stream("hi", conv3):
        pass
except Exception:
    pass
print("len:", len(conv3))