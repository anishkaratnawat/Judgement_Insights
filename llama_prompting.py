import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer,    pipeline
import pandas as pd


device_map='auto'

model_name = "NousResearch/Llama-2-7b-hf"

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,)


# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# Load your dataset
dataset_path = "/raid/home/anishkar/semeval/semeval-2023-legaleval/data/SeperatedData/train.csv"
valid_dataset = pd.read_csv(dataset_path)
csv_data = {}

texts = []
actual_labels = []
predicted_labels = []
index=40
for index, row in valid_dataset.iterrows():
    text = row['text']
    label = row['label']
    if len(text)>20000:
        continue
    if index > 1500:
        break
    

    prompt = f'Explain whether the following petition is accepted or rejected in binary classification. Petition: {text}'

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_label = torch.argmax(outputs.logits).item()

    print(index)
    print(predicted_label)
    

    texts.append(text)

    
    actual_labels.append(label)
    predicted_labels.append(predicted_label)

csv_data['text'] = texts
csv_data['label'] = actual_labels
csv_data['Predicted Label'] = predicted_labels

df = pd.DataFrame(csv_data)

df.to_csv("llama_prompting_results2.csv", index=False)
