import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from peft import PeftModel
import pandas as pd
from tqdm import tqdm
import evaluate

metric = evaluate.load("sacrebleu")

model_name = "Equall/Saul-Instruct-v1"
new_model = "/raid/home/anishkar/semeval/output_saul/checkpoint-7000"

device_map='auto'

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
model = PeftModel.from_pretrained(base_model, new_model)
model = model.merge_and_unload()

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

#Validate on the validation set
valid_dataset = pd.read_csv("/raid/home/anishkar/semeval/semeval-2023-legaleval/data/SeperatedData/train.csv")
saul_results = {"text": [], "label": [], "saul_generation": []}
print('Evaluating')
i=1
alpaca_prompt = """
### Instruction:
{}

### Input:
{}

### Response:
{}"""

for index, row in valid_dataset.iterrows():

    text = row['text']

    label = row['label']

    prompt = f'Explain whether the following petition is accepted or rejected.Petition:{text}'



    try:
        pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=50)
    except Exception as e:
        print("Error occurred during pipeline initialization:", e)

    result = pipe(f"<s>[INST] {prompt} [/INST]")

    generated = result[0]['generated_text']

    saul_results['text'].append(text)

    
    saul_results['label'].append(label)
    start = generated.find("[/INST]") + len("[/INST]") + 1

    end = generated.find("</s>")

    if end == -1:

        prediction = generated[start:]
    else:

        prediction = generated[start:end]


    
    saul_results['saul_generation'].append(prediction)

    print(f"Classifier Output: {generated}")
    print(i)
    i=i+1
    if index == 2:
        break

df = pd.DataFrame(saul_results)
df.to_csv('saul_results2.csv', index=False)
