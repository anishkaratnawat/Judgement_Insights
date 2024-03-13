import numpy as np
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import re
torch.cuda.empty_cache()

model = "NousResearch/Llama-2-7b-chat-hf"
# model = "meta-llama/Llama-2-7b-chat-hf"
save_dir = "models/llama-finetuned"
log_dir = "results/llama-finetuned"
dataset = "ildc"

# Fine-tuned model name
new_model = "Llama-2-7b-chat-finetune"

# LoRA attention dimension
lora_r = 64

# Alpha parameter for LoRA scaling
lora_alpha = 16

# Dropout probability for LoRA layers
lora_dropout = 0.1

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

# Output directory where the model predictions and checkpoints will be stored
output_dir = "./results"

# Number of training epochs
num_train_epochs = 1

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = False

# Batch size per GPU for training
per_device_train_batch_size = 4

# Batch size per GPU for evaluation
per_device_eval_batch_size = 4

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# Optimizer to use
optim = "paged_adamw_32bit"

# Learning rate schedule
lr_scheduler_type = "cosine"

# Number of training steps (overrides num_train_epochs)
max_steps = -1

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 0

# Log every X updates steps
logging_steps = 25

# Maximum sequence length to use
max_seq_length = None

device_map = {"": 0}

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False


train_data = []
dev_data = []
test_data = []
ildc_single = []
ildc_multi = []

if dataset == "ildc":
    print("Load dataset ILDC")
    # ildc_single = pd.read_csv("drive/MyDrive/semeval-dataset/ILDC_single/ILDC_single/ILDC_single.csv/ILDC_single.csv")
    # ildc_multi = pd.read_csv("drive/MyDrive/semeval-dataset/ILDC_multi/ILDC_multi/ILDC_multi.csv/ILDC_multi.csv")
    ildc_single = pd.read_csv("data/ILDC/ILDC_single/ILDC_single.csv")
    ildc_multi = pd.read_csv("data/ILDC/ILDC_multi/ILDC_multi.csv")

elif dataset == "semeval":
    print("Load dataset SemEval")
    # ildc_single = pd.read_csv("data/subtask3/ILDC_single_train_dev.csv")
    # ildc_multi = pd.read_csv("data/subtask3/ILDC_multi_train_dev.csv")

y_test = []
for text, label, split in np.concatenate(
    [ildc_single.values, ildc_multi.values], axis=0
)[:, :3]:
    text = text.replace("\n", " ")
    text = re.sub("\s+", " ", text).strip()

    text = text.lower()

    if split == "train":
        train_data.append({"text": "<s>[INST] " + text + " [/INST]" + str(label) + "</s>"})
    elif split == "dev":
        dev_data.append({"text": "<s>[INST] " + text + " [/INST]"})
    elif split == "test":
        y_test.append(str(label))
        test_data.append({"text":"<s>[INST] " + text + " [/INST]"})

if len(test_data) == 0:
    test_data = dev_data

device = "cuda" if torch.cuda.is_available() else "cpu"

# Creating dataset objects
train_dataset = Dataset.from_list(train_data[0: 1000])
dev_dataset = Dataset.from_list(dev_data)
test_dataset = Dataset.from_list(test_data)
print(train_dataset)

# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

# Load base model
model1 = AutoModelForCausalLM.from_pretrained(
    model,
    quantization_config=bnb_config,
    device_map=device_map
)
model1.config.use_cache = False
model1.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model1,
    train_dataset=train_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)


print("Start training")
trainer.train()
trainer.save_model(save_dir)


# Ignore warnings
# logging.set_verbosity(logging.CRITICAL)


pipe = pipeline(task="text-generation", model=model1, tokenizer=tokenizer, max_length=200)

predict_result = []
for prompt in test_data:
    result = pipe(prompt.text)
    predict_result.append(result[0]['generated_text'])

print(classification_report(y_test, predict_result))