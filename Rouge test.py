import os
import json
from glob import glob
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from rouge_score import rouge_scorer

# Define paths and load model/tokenizer
test_data_dir = r"C:\Harshil\VT-SSum-main\VT-SSum-main\test"
model_path = r"C:\Harshil\results\checkpoint-69"  # Replace with actual model path if needed

# Load the model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained(model_path).to("cuda")

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Function to evaluate each JSON file
def evaluate_json_file(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Extract input and target summaries
    segments = [" ".join(segment) for segment in data.get("segmentation", [])]
    input_text = " ".join(segments)

    summarization_data = data.get("summarization", {})
    target_summaries = [
        " ".join([sent["sent"] for sent in clip_data["summarization_data"] if sent["label"] == 1])
        for clip_key, clip_data in summarization_data.items() if clip_data.get("is_summarization_sample")
    ]

    # Skip files with no valid target summaries
    if not target_summaries:
        return []

    # Generate model summaries and compute ROUGE scores
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length").to("cuda")
    summary_ids = model.generate(inputs["input_ids"], max_length=128, num_beams=4, early_stopping=True)
    generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Calculate ROUGE for each target summary
    rouge_scores = []
    for target_summary in target_summaries:
        scores = scorer.score(target_summary, generated_summary)
        rouge_scores.append(scores)

    return rouge_scores

# Evaluate all files and aggregate results
all_rouge_scores = []
test_files = glob(os.path.join(test_data_dir, '*.json'))

for test_file in test_files:
    file_scores = evaluate_json_file(test_file)
    all_rouge_scores.extend(file_scores)

# Compute average ROUGE scores
average_scores = {}
for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
    average_scores[rouge_type] = {
        "precision": sum(score[rouge_type].precision for score in all_rouge_scores) / len(all_rouge_scores),
        "recall": sum(score[rouge_type].recall for score in all_rouge_scores) / len(all_rouge_scores),
        "fmeasure": sum(score[rouge_type].fmeasure for score in all_rouge_scores) / len(all_rouge_scores),
    }

print("Average ROUGE Scores:", average_scores)
