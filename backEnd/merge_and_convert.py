from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import subprocess

base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"

adapter_path = "/Users/duanegennaro/theTalkingMachine/v3_model/qwen-spoken-engagement-chatml-lora-final"
merged_path = "/Users/duanegennaro/theTalkingMachine/v3_model/qwen-spoken-engagement-chatml-merged"
mlx_output_path = "/Users/duanegennaro/theTalkingMachine/v3_model/qwen-spoken-engagement-chatml-mlx-4bit"

def run():
    if os.path.isdir(mlx_output_path):
        print(f"MLX model already exists at {mlx_output_path}, skipping merge/convert.")
        return

    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(f"Adapter directory not found at {adapter_path}")

    print("Loading base model (full precision, CPU/MPS -- no quantization needed for merge)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    print("Loading LoRA adapter and merging into base weights...")
    merged_model = PeftModel.from_pretrained(base_model, adapter_path).merge_and_unload()

    print(f"Saving merged model to {merged_path} ...")
    os.makedirs(merged_path, exist_ok=True)
    merged_model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)

    print(f"Converting merged model to MLX 4-bit format at {mlx_output_path} ...")
    subprocess.run(
        [
            "mlx_lm.convert",
            "--hf-path", merged_path,
            "-q",
            "--mlx-path", mlx_output_path,
        ],
        check=True,
    )

    print(f"\nDone. Point mlx_lm.load() at: {mlx_output_path}")

if __name__ == "__main__":
    run()