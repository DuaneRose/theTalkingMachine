import onnxruntime as ort
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
import numpy as np

model_id = "latishab/turnsense"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model_path = hf_hub_download(repo_id=model_id, filename="model_preprocessed.onnx")
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

def is_turn_complete(text: str) -> float:
    inputs = tokenizer(f"<|user|> {text}", padding="max_length", max_length=256, return_tensors="np")
    logits = session.run(None, {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
    })[0][0]
    probs = np.exp(logits) / np.sum(np.exp(logits))  # softmax
    return probs[1]  # probability the utterance is complete 