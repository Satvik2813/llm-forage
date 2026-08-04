import os
from dotenv import load_dotenv
from google import genai

# 1. Load environment variables from .env
load_dotenv()

# 2. Get Gemini API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

# 3. Create Gemini client
client = genai.Client(api_key=api_key)

# 4. Send request to Gemini
response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents="Say hi in 3 words"
)

# 5. Print response
print(response.text)