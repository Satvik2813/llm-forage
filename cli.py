import argparse
import time

from forge.client import LLMClient
from forge.config import ModelConfig
from forge.conversation import Conversation


def parse_args():
    parser = argparse.ArgumentParser(
        description="LLMForge interactive CLI"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="flash",
        help="Model name or alias (default: flash)",
    )

    parser.add_argument(
        "--temp",
        type=float,
        default=0.7,
        help="Temperature between 0 and 2 (default: 0.7)",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p between 0 and 1 (default: 1.0)",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum output tokens (default: 1024)",
    )

    parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="System instruction",
    )

    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming responses",
    )

    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Load a saved conversation",
    )

    return parser.parse_args()


def print_help():
    print(
        """
Commands:
  /help          Show available commands
  /cost          Show usage and cost summary
  /clear         Clear conversation history
  /save <name>   Save conversation
  /load <name>   Load conversation
  /tokens        Show message and token usage
  /config        Show current configuration
  /exit          Show session summary and exit
"""
    )


def normalize_save_path(name: str) -> str:
    name = name.strip()

    if not name:
        raise ValueError("Conversation name cannot be empty.")

    if not name.endswith(".json"):
        name += ".json"

    return name


def print_config(
    config: ModelConfig,
    stream: bool,
):
    print()
    print("Current configuration")
    print("---------------------")
    print(f"Model:       {config.model}")
    print(f"Temperature: {config.temperature}")
    print(f"Top-p:       {config.top_p}")
    print(f"Max tokens:  {config.max_tokens}")
    print(f"Streaming:   {stream}")
    print(f"System:      {config.system or 'None'}")
    print()


def print_tokens(
    llm: LLMClient,
    conv: Conversation,
):
    usage = llm.usage

    total_tokens = (
        usage.input_tokens
        + usage.output_tokens
    )

    print()
    print("Token information")
    print("-----------------")
    print(f"Conversation messages: {len(conv)}")
    print(f"API calls:             {usage.call_count}")
    print(f"Input tokens:          {usage.input_tokens:,}")
    print(f"Output tokens:         {usage.output_tokens:,}")
    print(f"Total tokens:          {total_tokens:,}")
    print()


def print_status(
    config: ModelConfig,
    llm: LLMClient,
    elapsed: float,
    previous_cost: float,
):
    current_cost = llm.usage.total_cost()

    request_cost = current_cost - previous_cost

    total_tokens = (
        llm.usage.input_tokens
        + llm.usage.output_tokens
    )

    model_name = config.model

    if model_name.startswith("gemini-"):
        model_name = model_name[len("gemini-"):]

    print(
        f"[{model_name} "
        f"· t={config.temperature} "
        f"· {total_tokens:,} tok "
        f"· {elapsed:.1f}s "
        f"· ${request_cost:.4f} "
        f"· session ${current_cost:.4f}]"
    )


def handle_command(
    command: str,
    llm: LLMClient,
    conv: Conversation,
    config: ModelConfig,
    stream: bool,
):
    parts = command.strip().split(maxsplit=1)

    cmd = parts[0].lower()

    argument = (
        parts[1].strip()
        if len(parts) > 1
        else None
    )

    if cmd == "/help":
        print_help()

        return conv, False

    if cmd == "/cost":
        print()
        print(llm.usage.summary())
        print()

        return conv, False

    if cmd == "/clear":
        conv.clear()

        print("Conversation cleared.")

        return conv, False

    if cmd == "/save":
        if not argument:
            print("Usage: /save <name>")

            return conv, False

        try:
            path = normalize_save_path(argument)

            conv.save(path)

            print(f"Conversation saved to {path}")

        except (OSError, ValueError) as e:
            print(f"Could not save conversation: {e}")

        return conv, False

    if cmd == "/load":
        if not argument:
            print("Usage: /load <name>")

            return conv, False

        try:
            path = normalize_save_path(argument)

            conv = Conversation.load(path)

            print(
                f"Loaded {path} "
                f"({len(conv)} messages)"
            )

        except (OSError, ValueError) as e:
            print(f"Could not load conversation: {e}")

        return conv, False

    if cmd == "/tokens":
        print_tokens(llm, conv)

        return conv, False

    if cmd == "/config":
        print_config(
            config=config,
            stream=stream,
        )

        return conv, False

    if cmd == "/exit":
        print()
        print("Session summary")
        print("===============")
        print(llm.usage.summary())

        return conv, True

    print(
        f"Unknown command: {cmd}\n"
        "Type /help to see available commands."
    )

    return conv, False


def main():
    args = parse_args()

    # ---------------------------------
    # Config + client
    # ---------------------------------

    try:
        config = ModelConfig(
            model=args.model,
            temperature=args.temp,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            system=args.system,
        )

        llm = LLMClient(config)

    except (ValueError, OSError) as e:
        print(f"Configuration error: {e}")
        return

    # ---------------------------------
    # Conversation
    # ---------------------------------

    if args.load:
        try:
            path = normalize_save_path(args.load)

            conv = Conversation.load(path)

            print(
                f"Loaded conversation: "
                f"{path} ({len(conv)} messages)"
            )

        except (OSError, ValueError) as e:
            print(
                f"Could not load conversation "
                f"'{args.load}': {e}"
            )
            return

    else:
        conv = Conversation()

    # ---------------------------------
    # Startup
    # ---------------------------------

    print()
    print("LLMForge")
    print("========")
    print(f"Model: {config.model}")

    if args.stream:
        print("Mode: streaming")
    else:
        print("Mode: normal")

    print("Type /help for commands.")
    print("Press Ctrl+C to cancel a response.")
    print()

    # ---------------------------------
    # Main loop
    # ---------------------------------

    while True:
        try:
            user_input = input("> ").strip()

        except KeyboardInterrupt:
            print()
            print("Use /exit to quit.")
            continue

        except EOFError:
            print()
            print()
            print("Session summary")
            print("===============")
            print(llm.usage.summary())
            break

        if not user_input:
            continue

        # ---------------------------------
        # Commands
        # ---------------------------------

        if user_input.startswith("/"):
            conv, should_exit = handle_command(
                command=user_input,
                llm=llm,
                conv=conv,
                config=config,
                stream=args.stream,
            )

            if should_exit:
                break

            continue

        # ---------------------------------
        # Model request
        # ---------------------------------

        previous_cost = llm.usage.total_cost()
        start_time = time.time()

        try:
            if args.stream:
                print()

                for chunk in llm.stream(
                    user_input,
                    conversation=conv,
                ):
                    print(
                        chunk,
                        end="",
                        flush=True,
                    )

                print()

            else:
                response = llm.chat(
                    user_input,
                    conversation=conv,
                )

                print()
                print(response)

        except KeyboardInterrupt:
            print()
            print()
            print("Response cancelled.")

            # client.stream() finally block
            # rolls back the orphan user message.
            continue

        except Exception as e:
            print()
            print(f"Error: {e}")
            continue

        # ---------------------------------
        # Status
        # ---------------------------------

        elapsed = time.time() - start_time

        print_status(
            config=config,
            llm=llm,
            elapsed=elapsed,
            previous_cost=previous_cost,
        )


if __name__ == "__main__":
    main()