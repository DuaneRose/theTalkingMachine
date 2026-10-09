import json
import threading
from ollama import chat
context_lock = threading.Lock()


file = "/Users/duanegennaro/theTalkingMachine/backEnd/context.json"

def get_context():
    try:
        with open(file, "r") as f:
            context_data = json.load(f)
    except FileNotFoundError:
        context_data = {"conversation": [], "summary": ""}

    return context_data

def add_context(new_context):
    with context_lock:
        try:
            with open(file, "r") as f:
                context_data = json.load(f)
        except FileNotFoundError:
            context_data = {"conversation": [], "summary": ""}
        except json.JSONDecodeError:
            context_data = {"conversation": [], "summary": ""}

        context_data["conversation"].append(new_context)

        should_summarize = len(context_data["conversation"]) >= 15

        with open(file, "w") as f:
            json.dump(context_data, f, indent=4)

    if should_summarize:
        summerize = threading.Thread(target=summarize_context)
        summerize.start()

def clear_context():
    with context_lock:
        context_data = {"conversation": [], "summary": ""}
        with open(file, "w") as f:
            json.dump(context_data, f, indent=4)

def summarize_context():
    try:
        with open(file, "r") as f:
            context_data = json.load(f)
    except FileNotFoundError:
        context_data = {"conversation": [], "summary": ""}

    to_summarize = context_data["conversation"]
    previous_summary = context_data.get("summary", "")

    summary = chat(
        model="qwen3:4b-instruct",
        messages=[
            {
                "role": "system",
                "content": (
                    "You extract literal facts from a conversation. Write only what was "
                    "explicitly said, using neutral third-person statements. Do not "
                    "interpret intent, emotional state, goals, or themes. Do not use "
                    "abstract language like 'energy', 'process', 'performance', or "
                    "'protecting' anything, unless the user used that exact word.\n\n"
                    "Example of BAD (interpretive): 'The user is working through "
                    "feelings about productivity and self-worth.'\n"
                    "Example of GOOD (literal): 'The user said they plan to work on "
                    "their project and will use a timer.'\n\n"
                    "4-7 plain sentences max. No headers, no lists."
                ),
            },
            {
                "role": "user",
                "content": f"""Previous memory: {previous_summary or "(none yet)"}

                New turns to fold in:
                {to_summarize}

                Write the updated memory note."""
            }
        ]
    )

    context_data["conversation"] = to_summarize[-6:]
    context_data["summary"] = summary["message"]["content"]

    with open(file, "w") as f:
        json.dump(context_data, f, indent=4)

    return "Context summarized and saved."