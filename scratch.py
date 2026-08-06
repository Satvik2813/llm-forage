import time
from google import genai
from dotenv import load_dotenv
import os

from forge.config import ModelConfig
from forge.client import LLMClient

load_dotenv()

print("=" * 50)
print("1. AVAILABLE MODELS")
print("=" * 50)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
for m in client.models.list():
    if "generateContent" in getattr(m, "supported_actions", []):
        print(m.name)

print()
print("=" * 50)
print("2. CONFIG")
print("=" * 50)
c = ModelConfig(model="flash", temperature=0.7)
print("model:", c.model)
print("config:", c.to_gemini_config())

c2 = ModelConfig(model="flash", system="You are a tutor")
print("with system:", c2.to_gemini_config())

try:
    ModelConfig(model="flash", temperature=5.0)
    print("BUG: validation failed")
except ValueError as e:
    print("validation OK:", e)

print()
print("=" * 50)
print("3. CHAT")
print("=" * 50)
llm = LLMClient(ModelConfig(model="flash"))
print(llm.chat("Say hi in exactly 3 words"))

print()
print("=" * 50)
print("4. STREAM")
print("=" * 50)
for chunk in llm.stream("Count from 1 to 10"):
    print(chunk, end="", flush=True)
print()
print("last_response:", repr(llm.last_response))

print()
print("=" * 50)
print("5. INVALID KEY (should fail INSTANTLY)")
print("=" * 50)
start = time.time()
try:
    LLMClient(ModelConfig(model="flash"), api_key="invalid_123").chat("hi")
    print("BUG: no error raised")
except Exception as e:
    print("type:", type(e).__name__)
    print("code:", getattr(e, "code", "NO CODE ATTR"))
    print("elapsed:", round(time.time() - start, 2), "s")