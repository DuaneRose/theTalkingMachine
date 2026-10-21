from context_manager import add_context, get_context
from print_color import print_blue
from ollama import chat

brain = "qwen3.5:4b"

def communicate(text):
    context = get_context() or {
        "summary": "",
        "conversation": []
    }

    # Keep this only if your audio pipeline appends a 25-character marker.
    user_text = text[:-25] if len(text) > 25 else text

    messages = [
        {
            "role": "system",
            "content": (
                "You are a warm, concise conversational assistant.\n"
                "The conversation history below is background context only.\n"
                "The final user message is the current request and has priority.\n"
                "Answer the final user message directly.\n"
                "Do not continue or answer an older topic unless the final message asks you to.\n"
                "If the final message is unclear or appears to be speech-recognition noise, "
                "ask the user to repeat themselves."
                "Summary of earlier conversation: \n\n"
                f"{context.get('summary', '')}"
            ),
        }
    ]

    # Add previous conversation history
    for turn in context.get("conversation", []):
        messages.append({
            "role": "user",
            "content": turn["user"]
        })

        messages.append({
            "role": "assistant",
            "content": turn["assistant"]
        })

    # Add the current user message
    messages.append({
        "role": "user",
        "content": user_text
    })

    stream = chat(
        model=brain,
        messages=messages,
        stream=True,
        think=False
    )

    response_text = ""

    print_blue(messages)

    for chunk in stream:
        piece = chunk.message.content

        if piece:
            response_text += piece
            yield piece

    # Save the completed exchange after streaming finishes
    add_context({
        "user": text,
        "assistant": response_text
    })