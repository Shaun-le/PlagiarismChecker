import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import re
from datasets import Dataset

tokenizer = AutoTokenizer.from_pretrained('PlagiarismDetectionModel')
model = AutoModelForSequenceClassification.from_pretrained("PlagiarismDetectionModel", num_labels=2)
def preprocess_function(examples):
    model_inputs = tokenizer(examples["origins"], examples["generations"], truncation=True)
    return model_inputs

def detection(s1: str, s2: str):
    new_data = {'origins': [f"{s1}"],
                'generations': [f"{s2}"]}
    new_dataset = Dataset.from_dict(new_data)

    tokenized_new_data = new_dataset.map(preprocess_function, batched=True, remove_columns=['origins', 'generations'], num_proc=1)

    input_ids = torch.tensor(tokenized_new_data["input_ids"])
    attention_mask = torch.tensor(tokenized_new_data["attention_mask"])
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    predicted_labels = np.argmax(outputs.logits.detach().numpy(), axis=1)

    return predicted_labels[0]

def find_candiate(cont):
    df = pd.read_csv('datasets/results.csv')
    #df["prediction"] = df["prediction"].apply(lambda x: ''.join(eval(x)))
    new_data = {'filename': [],
                'prediction': []}
    for i in range(len(df)):
        prediction = detection(cont, ''.join(eval(df['prediction'][i])))
        if prediction == 1:
            new_data['filename'].append(df['filename'][i])
            new_data['prediction'].append(df['prediction'][i])

    new_df = pd.DataFrame(new_data)
    new_df.to_csv('datasets/candidate.csv', index=False)
    return new_df

