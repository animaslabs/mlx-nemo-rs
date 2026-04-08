#!/usr/bin/env python3
"""Interactive chat with an LLM using mlx_lm."""

import sys
from mlx_lm import load, stream_generate

MODEL = "mlx-community/GLM-4.7-Flash-4bit"


def chat(model, tokenizer, messages: list[dict], user_input: str) -> str:
    """Send a message and get a response, streaming output as it generates."""
    messages.append({"role": "user", "content": user_input})

    if tokenizer.chat_template is not None:
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_dict=False
        )
    else:
        prompt = user_input

    response_chunks = []
    for chunk in stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=1024,
    ):
        print(chunk.text, end="", flush=True)
        response_chunks.append(chunk.text)

    print()  # newline after response
    response = "".join(response_chunks)
    messages.append({"role": "assistant", "content": response})
    return response


def interactive_mode(model, tokenizer):
    """Run interactive chat loop."""
    print("Model loaded. Type 'quit' or 'exit' to end the conversation.\n")
    messages = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        print("Assistant: ", end="", flush=True)
        chat(model, tokenizer, messages, user_input)
        print()  # extra blank line between turns


def main():
    print("Loading model...")
    model, tokenizer = load(MODEL)

    if len(sys.argv) > 1:
        # Single message mode from command line
        user_input = " ".join(sys.argv[1:])
        messages = []
        print("Assistant: ", end="", flush=True)
        chat(model, tokenizer, messages, user_input)
    else:
        # Interactive mode
        interactive_mode(model, tokenizer)


if __name__ == "__main__":
    main()
