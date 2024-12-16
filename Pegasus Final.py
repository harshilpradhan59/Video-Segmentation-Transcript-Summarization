import os
import json
from glob import glob
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import torch

if not torch.cuda.is_available():
    print("GPU not available. Make sure to configure a compatible GPU environment.")
else:
    print("GPU is available!")

data_dir = r"C:\Harshil\VT-SSum-main\VT-SSum-main\train"
all_files = glob(os.path.join(data_dir, '*.json'))

tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum").to("cuda")

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

print(f"Total data examples loaded: {len(all_data)}")
if len(all_data) == 0:
    raise ValueError("No data found in the JSON files. Check the preprocessing logic or file paths.")

dataset = Dataset.from_dict({
    "input_text": [item["input_text"] for item in all_data],
    "summary_text": [item["summary_text"] for item in all_data]
})

train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

print(f"Train Dataset Size: {len(train_dataset)}")
print(f"Eval Dataset Size: {len(eval_dataset)}")

if len(train_dataset) == 0:
    raise ValueError("Train dataset is empty. Check the splitting logic.")
if len(eval_dataset) == 0:
    raise ValueError("Eval dataset is empty. Check the splitting logic.")

def tokenize_data(example):
    input_encodings = tokenizer(example["input_text"], truncation=True, padding="max_length", max_length=512)
    target_encodings = tokenizer(example["summary_text"], truncation=True, padding="max_length", max_length=128)
    return {
        "input_ids": input_encodings["input_ids"],
        "attention_mask": input_encodings["attention_mask"],
        "labels": target_encodings["input_ids"]
    }

train_dataset = train_dataset.map(
    lambda x: tokenize_data(x),
    batched=True,
    remove_columns=["input_text", "summary_text"]
)
eval_dataset = eval_dataset.map(
    lambda x: tokenize_data(x),
    batched=True,
    remove_columns=["input_text", "summary_text"]
)

print("Tokenized train dataset example:", train_dataset[0])
print("Tokenized eval dataset example:", eval_dataset[0])

train_dataset.set_format("torch")
eval_dataset.set_format("torch")

training_args = TrainingArguments(
    output_dir="./results/pegasus",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs/pegasus",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    fp16=torch.cuda.is_available(),
)

torch.cuda.empty_cache()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()

metrics = trainer.evaluate()
print("Evaluation Metrics:", metrics)

trainer.save_model("./results/pegasus-final")
print("Model saved successfully!")
