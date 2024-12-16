from transformers import BartForConditionalGeneration, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset
import torch
from glob import glob
import os
import json
from transformers import BartTokenizer

if not torch.cuda.is_available():
    print("GPU not available. Make sure to configure a compatible GPU environment.")
else:
    print("GPU is available!")

data_dir = r"C:\Harshil\VT-SSum-main\VT-SSum-main\train"
all_files = glob(os.path.join(data_dir, '*.json'))

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large").to("cuda")

def preprocess_file(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    segments = [" ".join(segment) for segment in data.get("segmentation", [])]
    input_text = " ".join(segments)
    summaries = []
    summarization_data = data.get("summarization", {})
    for clip_key, clip_data in summarization_data.items():
        if clip_data.get("is_summarization_sample"):
            summary = " ".join([sent["sent"] for sent in clip_data["summarization_data"] if sent["label"] == 1])
            if summary:
                summaries.append({"input_text": input_text, "summary_text": summary})
    return summaries

all_data = []
for file in all_files:
    all_data.extend(preprocess_file(file))

dataset = Dataset.from_dict({
    "input_text": [item["input_text"] for item in all_data],
    "summary_text": [item["summary_text"] for item in all_data]
})

train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

def tokenize_data(example):
    input_encodings = tokenizer(example["input_text"], truncation=True, padding="max_length", max_length=512)
    target_encodings = tokenizer(example["summary_text"], truncation=True, padding="max_length", max_length=128)
    return {"input_ids": input_encodings["input_ids"], "attention_mask": input_encodings["attention_mask"], "labels": target_encodings["input_ids"]}

train_dataset = train_dataset.map(tokenize_data, batched=True)
eval_dataset = eval_dataset.map(tokenize_data, batched=True)

train_dataset = train_dataset.remove_columns(["input_text", "summary_text"])
eval_dataset = eval_dataset.remove_columns(["input_text", "summary_text"])

train_dataset.set_format("torch")
eval_dataset.set_format("torch")

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    fp16=True,
    logging_dir="./logs",
    save_steps=500,
    eval_steps=500,
    gradient_accumulation_steps=4,
    load_best_model_at_end=True,
)

early_stopping = EarlyStoppingCallback(early_stopping_patience=2)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[early_stopping]
)

trainer.train()

metrics = trainer.evaluate()
print("Evaluation Metrics:", metrics)
