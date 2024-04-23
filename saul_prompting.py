import torch
from transformers import pipeline
import pandas as pd

# Define your model and tokenizer
model_name = "Equall/Saul-Instruct-v1"
pipe = pipeline("text-generation", model=model_name, torch_dtype=torch.bfloat16, device_map="cpu")

# Function to generate prompt for each document
def generate_prompt(document):
    messages = [
        {"role": "user", "content": f"{document} Explain whether this petition is accepted or rejected."},
    ]
    return messages
    # return f"{document} Explain whether this petition is accepted or rejected."

# Function to classify document, provide reasons, and influential sentences
def classify_document(document):
    # Generate prompt
    prompt = generate_prompt(document)
    
    # Get model outputs
    outputs = pipe(prompt, max_new_tokens=256, do_sample=False)
    
    # Extract classifier output
    classifier_output = outputs[0]["generated_text"]
    
    
    return classifier_output

# Read data from CSV file
data = pd.read_csv("/data2/home/bhavyac/NLP_Project/train.csv")
csv_data = {}

texts = []
actual_labels = []
predicted_labels = []

# Loop through each document in the dataset
for index, row in data.iterrows():
    document = row['text']
    label = row['label']

    if index == 200:
        break
    
    # Call classify_document function
    classifier_output = classify_document(document)
    
    # Print results
    print(f"Document: {index}")
    print(f"Actual Label: {label}")
    
    for i in range(len(classifier_output)):
        if classifier_output[i]['role'] == 'assistant':
            texts.append(document)
            actual_labels.append(label)
            predicted_labels.append(classifier_output[i]['content'])
            print(f"Classifier Output: {classifier_output[i]['content']}")
    

    print("------------------------------")

# To Download predictions as CSV
csv_data['text'] = texts
csv_data['label'] = actual_labels
csv_data['Predicted Label'] = predicted_labels

df = pd.DataFrame(csv_data)

df.to_csv("/data2/home/bhavyac/NLP_Project/saul_prompting_results.csv", index=False)