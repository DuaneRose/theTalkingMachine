import os
import time
import json
from datetime import datetime

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


# -----------------------------
# Simple logging helpers
# -----------------------------
START_TIME = time.time()


def log(message):
    elapsed = time.time() - START_TIME
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now} | +{elapsed:7.1f}s] {message}", flush=True)


def section(title):
    print("\n" + "=" * 80, flush=True)
    log(title)
    print("=" * 80, flush=True)


def require_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            "Make sure train.py and both JSONL dataset files are uploaded to the same Colab folder."
        )
    size_mb = os.path.getsize(path) / (1024 * 1024)
    log(f"Found {path} ({size_mb:.2f} MB)")


class ProgressCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        log(f"Training started. Max steps: {state.max_steps}, epochs: {args.num_train_epochs}")

    def on_epoch_begin(self, args, state, control, **kwargs):
        log(f"Starting epoch {state.epoch if state.epoch is not None else 0:.2f}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            clean_logs = {k: round(v, 5) if isinstance(v, float) else v for k, v in logs.items()}
            log(f"Trainer log at step {state.global_step}: {clean_logs}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            clean_metrics = {k: round(v, 5) if isinstance(v, float) else v for k, v in metrics.items()}
            log(f"Evaluation finished at step {state.global_step}: {clean_metrics}")

    def on_save(self, args, state, control, **kwargs):
        log(f"Checkpoint saved at step {state.global_step} in {args.output_dir}")

    def on_train_end(self, args, state, control, **kwargs):
        log(f"Training finished at global step {state.global_step}")


# -----------------------------
# Config
# -----------------------------
section("Starting Qwen2.5 active-conversation LoRA training")

model_id = "Qwen/Qwen2.5-1.5B-Instruct"

train_file = "qwen2_5_spoken_engagement_chatml_v4_train.jsonl"
val_file = "qwen2_5_spoken_engagement_chatml_v4_val.jsonl"

output_dir = "./qwen-spoken-engagement-chatml-lora"
final_dir = "./qwen-spoken-engagement-chatml-lora-final"

log(f"Model: {model_id}")
log(f"Train file: {train_file}")
log(f"Validation file: {val_file}")
log(f"Output dir: {output_dir}")
log(f"Final adapter dir: {final_dir}")

section("Checking files and environment")
require_file(train_file)
require_file(val_file)

log(f"PyTorch version: {torch.__version__}")
log(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU not detected. In Colab, go to Runtime -> Change runtime type -> GPU, "
        "then rerun the notebook. This script uses bitsandbytes 4-bit QLoRA, which expects CUDA."
    )

log(f"GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    total_gb = props.total_memory / (1024 ** 3)
    log(f"GPU {i}: {torch.cuda.get_device_name(i)} ({total_gb:.2f} GB VRAM)")

use_bf16 = False
compute_dtype = torch.float16
log(f"Using bf16: {use_bf16}")
log(f"4-bit compute dtype: {compute_dtype}")

# -----------------------------
# Dataset
# -----------------------------
section("Loading dataset")

data_files = {
    "train": train_file,
    "validation": val_file,
}

dataset = load_dataset("json", data_files=data_files)
log(str(dataset))
log(f"Train rows: {len(dataset['train'])}")
log(f"Validation rows: {len(dataset['validation'])}")

first_example = dataset["train"][0]
log("First raw example:")
print(json.dumps(first_example, indent=2)[:2000], flush=True)

# -----------------------------
# Tokenizer
# -----------------------------
section("Loading tokenizer")

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    log("Tokenizer had no pad token; set pad_token = eos_token")
else:
    log(f"Tokenizer pad token: {tokenizer.pad_token!r}")

log(f"Tokenizer vocab size: {len(tokenizer)}")

# -----------------------------
# Format data
# -----------------------------
section("Formatting dataset with Qwen chat template")


def format_example(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


formatted_dataset = dataset.map(
    format_example,
    remove_columns=dataset["train"].column_names,
)

preview_text = formatted_dataset["train"][0]["text"]
log("First formatted training example preview:")
print(preview_text[:1200], flush=True)

log("Measuring token lengths for a quick sanity check...")
sample_size = min(100, len(formatted_dataset["train"]))
token_lengths = []
for i in range(sample_size):
    token_lengths.append(len(tokenizer(formatted_dataset["train"][i]["text"])["input_ids"]))

log(
    f"Token length sample: min={min(token_lengths)}, "
    f"avg={sum(token_lengths) / len(token_lengths):.1f}, max={max(token_lengths)} "
    f"over {sample_size} examples"
)

# -----------------------------
# Model
# -----------------------------
section("Loading 4-bit base model")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

log("Downloading/loading model. This can take a few minutes the first time...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)

model.config.use_cache = False
log("Model loaded successfully")
log(f"Model device map: {getattr(model, 'hf_device_map', 'not available')}")

allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
log(f"CUDA memory after model load: allocated={allocated_gb:.2f} GB, reserved={reserved_gb:.2f} GB")

# -----------------------------
# Prepare for k-bit training
# -----------------------------
section("Preparing quantized model for k-bit training")

model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)
model.enable_input_require_grads()

log("Model prepared for k-bit training (LayerNorm cast to fp32, input grads enabled)")

# -----------------------------
# LoRA
# -----------------------------
section("Configuring LoRA")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

log(f"LoRA config: r={lora_config.r}, alpha={lora_config.lora_alpha}, dropout={lora_config.lora_dropout}")
log(f"LoRA target modules: {lora_config.target_modules}")

# -----------------------------
# Training args
# -----------------------------
section("Building trainer")

training_args = SFTConfig(
    output_dir=output_dir,

    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,

    num_train_epochs=3,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",

    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",

    fp16=True,
    bf16=False,

    dataset_text_field="text",
    max_length=1024,

    packing=False,
    report_to="none",
    gradient_checkpointing=True,
)

log("Training arguments summary:")
log(f"  batch size per device: {training_args.per_device_train_batch_size}")
log(f"  grad accumulation: {training_args.gradient_accumulation_steps}")
log(f"  effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
log(f"  epochs: {training_args.num_train_epochs}")
log(f"  learning rate: {training_args.learning_rate}")
log(f"  max length: {training_args.max_length}")
log(f"  fp16: {training_args.fp16}, bf16: {training_args.bf16}")

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=formatted_dataset["train"],
    eval_dataset=formatted_dataset["validation"],
    peft_config=lora_config,
    callbacks=[ProgressCallback()],
)

log("Trainer created successfully")
try:
    trainer.model.print_trainable_parameters()
except Exception as exc:
    log(f"Could not print trainable parameter summary: {exc}")

# -----------------------------
# Dtype diagnostic + forced fix
# -----------------------------
section("Checking trainable parameter dtypes")

dtype_counts = {}
bf16_trainable_names = []
for name, param in trainer.model.named_parameters():
    if param.requires_grad:
        dtype_counts[param.dtype] = dtype_counts.get(param.dtype, 0) + 1
        if param.dtype == torch.bfloat16:
            bf16_trainable_names.append(name)

log(f"Trainable parameter dtype breakdown: {dtype_counts}")

if bf16_trainable_names:
    log(
        f"Found {len(bf16_trainable_names)} trainable bf16 parameter(s), "
        f"e.g. {bf16_trainable_names[:5]} -- force-casting to float32 "
        f"(GradScaler requires fp32 master weights for trainable params under fp16 training)"
    )
    with torch.no_grad():
        for name, param in trainer.model.named_parameters():
            if param.requires_grad and param.dtype == torch.bfloat16:
                param.data = param.data.to(torch.float32)
    log("Force-cast complete")
else:
    log("No bf16 trainable parameters found -- dtypes are consistent")

# Also catch any trainable params that ended up fp16 -- GradScaler forbids this too
fp16_trainable_names = [
    name for name, param in trainer.model.named_parameters()
    if param.requires_grad and param.dtype == torch.float16
]
if fp16_trainable_names:
    log(
        f"Found {len(fp16_trainable_names)} trainable fp16 parameter(s), "
        f"e.g. {fp16_trainable_names[:5]} -- force-casting to float32 as well"
    )
    with torch.no_grad():
        for name, param in trainer.model.named_parameters():
            if param.requires_grad and param.dtype == torch.float16:
                param.data = param.data.to(torch.float32)
    log("Force-cast complete")

final_dtype_counts = {}
for name, param in trainer.model.named_parameters():
    if param.requires_grad:
        final_dtype_counts[param.dtype] = final_dtype_counts.get(param.dtype, 0) + 1
log(f"Final trainable parameter dtype breakdown: {final_dtype_counts}")

# -----------------------------
# Train
# -----------------------------
section("Starting training")

train_result = trainer.train()
log("Raw train result:")
print(train_result, flush=True)

# -----------------------------
# Save
# -----------------------------
section("Saving final LoRA adapter")

trainer.save_model(final_dir)
tokenizer.save_pretrained(final_dir)

log(f"Saved final LoRA adapter to: {final_dir}")
log("Files saved:")
for name in sorted(os.listdir(final_dir)):
    path = os.path.join(final_dir, name)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    log(f"  {name} ({size_mb:.2f} MB)")

section("Done")
log("Training script completed successfully")