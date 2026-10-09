from mlx_lm import load, stream_generate
import numpy as np
import sounddevice as sd
from context_manager import add_context, get_context

# Fine-tuned model: merged LoRA adapter + converted to MLX 4-bit.
# Produced by merge_and_convert.py -- run that first if this path doesn't exist yet.
brain, tokenizer = load(
    "/Users/duanegennaro/theTalkingMachine/v1_model/qwen-active-conversation-mlx-4bit"
)

# mlx_lm.convert doesn't always carry chat_template.jinja into the converted
# output directory, so load it directly from the adapter folder we know has it.
_CHAT_TEMPLATE_PATH = (
    "/Users/duanegennaro/theTalkingMachine/v1_model/qwen-active-conversation-lora-final/chat_template.jinja"
)

with open(_CHAT_TEMPLATE_PATH, "r") as f:
    CHAT_TEMPLATE = f.read()

def communicate(text):
    context = get_context()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a warm, concise conversational assistant. Sound natural and engaged. "
                "Keep replies short but useful. Ask one natural follow-up when it helps keep the conversation going.\n\n"
                f"Summary of earlier conversation: {context['summary']}"
            ),
        }
    ]

    # flatten conversation history into proper turns, in order, BEFORE the new question
    for turn in context["conversation"]:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})

    # current turn goes last
    messages.append({"role": "user", "content": text[:-25]})

    prompt = tokenizer.apply_chat_template(
        messages,
        chat_template=CHAT_TEMPLATE,
        tokenize=False,
        add_generation_prompt=True,
    )

    resp = ""
    for response in stream_generate(
        brain,
        tokenizer,
        prompt=prompt,
        max_tokens=80
    ):
        resp += response.text
        yield response.text

    context = {
        "user" : text,
        "assistant" : resp
    }
    add_context(context)

