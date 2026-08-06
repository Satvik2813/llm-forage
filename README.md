# LLMForge

> A clean, resilient, and stateful abstraction for interacting with Large Language Models.

LLMForge provides a robust, decoupled interface for interacting with LLM providers. Instead of scattering provider-specific SDK logic (like `google-genai` code) throughout an application, LLMForge encapsulates the complexity inside a single client component. This cleanly separates configuration, state management, usage tracking, and the interactive CLI from the underlying API integration, paving the way for easier multi-provider support in the future.

## Features

- **Stateful Conversations**: Retain conversation history automatically across multiple turns.
- **Stateless Chat**: Send one-off messages without affecting conversation memory.
- **Streaming Responses**: Real-time token streaming for a responsive user experience.
- **Resilient Retry Handling**: Automatic retries with exponential backoff for transient API errors.
- **Conversation Persistence**: Save and load conversation history to and from JSON files.
- **Safe Rollbacks**: Failed or cancelled API requests automatically remove the orphan user message to prevent conversation history corruption.
- **Usage & Cost Tracking**: Built-in tracking of token usage (input/output) and estimated API costs.
- **Model Configuration**: Clean configuration structure with support for system instructions and common generation parameters (temperature, top-p, max tokens).
- **Interactive CLI**: A feature-rich command-line interface for chatting, managing state, and viewing usage metrics.

## Project Structure

```text
LLMForage/
├── cli.py               # Interactive command-line interface
├── forge/
│   ├── __init__.py
│   ├── client.py        # LLM client with SDK integration and retry logic
│   ├── config.py        # Configuration and parameter validation
│   ├── conversation.py  # Conversation state and persistence
│   └── metrics.py       # Token usage and cost tracking
├── .env                 # Environment variables (not tracked in git)
└── requirements.txt     # Python dependencies
```

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Satvik2813/llm-forage.git
   cd llm-forage
   ```

2. **Create and activate a virtual environment (Windows PowerShell):**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your environment:**
   Create a `.env` file in the root directory and add your API key:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

## Library Usage

### Basic Conversation

```python
from forge import LLMClient, ModelConfig
from forge.conversation import Conversation

# Configure the model
config = ModelConfig(model="flash")
llm = LLMClient(config)

# Initialize a stateful conversation
conv = Conversation()

# The conversation history is maintained automatically
print(llm.chat("My name is Adithya.", conv))
print(llm.chat("What's my name?", conv))
```

### Streaming Responses

To stream tokens as they are generated:

```python
for chunk in llm.stream("Tell me a story.", conversation=conv):
    print(chunk, end="", flush=True)
```

### Usage & Cost Tracking

You can inspect the token usage and estimated cost of your session at any time:

```python
print(llm.usage.summary())
```

## CLI Usage

Start the interactive CLI:

```bash
python cli.py
```

Start the CLI with specific parameters and streaming enabled:

```bash
python cli.py --model flash --temp 0.5 --stream --system "You are a terse assistant."
```

### CLI Commands

| Command | Description |
|---|---|
| `/help` | Show available commands |
| `/cost` | Show usage and estimated cost |
| `/clear` | Clear conversation history |
| `/save <name>` | Save conversation to JSON |
| `/load <name>` | Load a saved conversation |
| `/tokens` | Show message/token usage |
| `/config` | Show current configuration |
| `/exit` | Print session summary and exit |

### Example CLI Session

```text
> python cli.py

LLMForge
========
Model: gemini-3.5-flash
Mode: normal
Type /help for commands.
Press Ctrl+C to cancel a response.

> Hello! What can you do?

I can answer questions, help write code, and much more! How can I assist you?
[gemini-3.5-flash · t=0.7 · 28 tok · 1.2s · $0.0001 · session $0.0001]

> /cost

Usage Summary
-------------
Calls: 1
Input tokens: 7
Output tokens: 21
Total tokens: 28
Estimated cost: $0.000199

By model:
  gemini-3.5-flash: 1 calls, 7 input, 21 output, $0.000199

> /save chat_1
Conversation saved to chat_1.json

> /exit

Session summary
===============
...
```

## Architecture

LLMForge is designed with a strict separation of concerns to decouple the application layer from the provider-specific SDK implementation:

```text
   CLI (cli.py)
        ↓
    LLMClient
        ↓
Provider SDK (google-genai)
```

- `cli.py` contains **no direct SDK imports**. It relies entirely on the `forge` abstractions.
- Provider-specific SDK interaction is strictly isolated inside `forge/client.py`.
- `Conversation` owns conversation state management and persistence independent of the client.
- `UsageTracker` owns usage accounting and cost calculations.
- `ModelConfig` handles generation parameter configuration and standardizes model names.

This architectural decision keeps the CLI layer clean and makes integrating alternative or additional LLM providers significantly easier in the future.

## Reliability

LLMForge is built to handle the realities of network communication:
- **Automatic Retries:** API calls that fail with transient errors (like rate limits or server errors) are automatically retried.
- **Exponential Backoff:** Retries are spaced out to avoid overwhelming the provider.
- **Smart Streaming:** Streaming requests are never retried once output has started, guaranteeing that no duplicate text is yielded to the application.
- **Conversation Consistency:** If a request fails entirely or is manually cancelled by the user, the corresponding user message is immediately rolled back. This prevents orphan user messages from corrupting the conversation context.

## Future Improvements

- Support for additional LLM providers (e.g., OpenAI, Anthropic).
- Comprehensive automated test suite.
- Richer token tracking and metrics visualization.
- Configurable storage backends for conversation persistence (e.g., SQLite).
