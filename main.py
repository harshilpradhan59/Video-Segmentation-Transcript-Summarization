# main.py
# A complete, single-file implementation for the Video Lecture Summarization project.
# Author: Harshil Pradhan (Project Concept)
# AI-Generated Code by Google Gemini

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5ForConditionalGeneration, T5Tokenizer,
    PegasusForConditionalGeneration, PegasusTokenizer,
    BartForConditionalGeneration, BartTokenizer,
    AdamW,
    get_scheduler
)
from tqdm.auto import tqdm
import evaluate # Hugging Face's library for metrics like ROUGE
import time
import datetime

# --- Configuration ---
# All hyperparameters and settings are centralized here for easy modification.
CONFIG = {
    "models": {
        "t5": {
            "model_class": T5ForConditionalGeneration,
            "tokenizer_class": T5Tokenizer,
            "pretrained_name": "t5-small" # Using small for faster demonstration
        },
        "pegasus": {
            "model_class": PegasusForConditionalGeneration,
            "tokenizer_class": PegasusTokenizer,
            "pretrained_name": "google/pegasus-xsum" # A popular summarization model
        },
        "bart": {
            "model_class": BartForConditionalGeneration,
            "tokenizer_class": BartTokenizer,
            "pretrained_name": "facebook/bart-large-cnn" # Strong summarization baseline
        }
    },
    "data": {
        "dummy_data_path": "dummy_vt_ssum.json",
        "max_input_length": 512,  # As specified in the report
        "max_target_length": 128 # As specified in the report
    },
    "training": {
        "num_epochs": 3, # Report mentions 15, using 3 for a quicker demo
        "train_batch_size": 4, # Report mentions 8, using 4 to reduce memory for demo
        "eval_batch_size": 4,
        "learning_rate": 3e-5, # As specified in the report
        "weight_decay": 0.01,
        "gradient_accumulation_steps": 2, # Report mentions 4, using 2 for demo
        "patience": 2 # For early stopping
    },
    "output_dir": "models"
}

# --- 1. Dummy Data Generation ---
# This function creates a dummy dataset to make the script runnable.
def create_dummy_dataset():
    """
    Creates a dummy JSON file mimicking the VT-SSum dataset structure.
    This allows the script to run without needing the actual dataset.
    """
    dummy_data = [
        {
            "id": "video1",
            "title": "Introduction to Deep Learning",
            "segmentation": [
                "Deep learning is a subset of machine learning based on artificial neural networks.",
                "The 'deep' in deep learning refers to the number of layers in the network.",
                "Today we will cover the basics of perceptrons and activation functions.",
                "We will also discuss the history of neural networks, from the early days to modern transformers."
            ],
            "summarization": {
                "is_summarization_sample": True,
                "summary_text": "This lecture introduces the fundamentals of deep learning, including its definition as a subset of machine learning, the concept of network depth, and the basic components like perceptrons and activation functions."
            }
        },
        {
            "id": "video2",
            "title": "Understanding Transformer Models",
            "segmentation": [
                "The transformer architecture was introduced in the paper 'Attention Is All You Need'.",
                "It relies heavily on a mechanism called self-attention, which allows the model to weigh the importance of different words in the input sequence.",
                "Unlike RNNs, transformers can process sequences in parallel, making them highly efficient for training on large datasets.",
                "Key components include the encoder, the decoder, positional encodings, and multi-head attention."
            ],
            "summarization": {
                "is_summarization_sample": True,
                "summary_text": "The lecture explains the transformer architecture from the 'Attention Is All You Need' paper. It highlights key features like self-attention, parallel processing capabilities, and its main components such as the encoder-decoder structure and multi-head attention."
            }
        },
        {
            "id": "video3",
            "title": "Data Preprocessing Techniques",
            "segmentation": [
                "Data preprocessing is a crucial step in any machine learning project.",
                "Techniques include handling missing values, data normalization, and feature scaling.",
                "For natural language processing, this also involves tokenization, stop-word removal, and stemming or lemmatization.",
                "Proper preprocessing can significantly improve model performance and training stability."
            ],
            "summarization": {
                "is_summarization_sample": False, # Example of a non-summary sample
                "summary_text": ""
            }
        }
    ]
    if not os.path.exists(CONFIG["data"]["dummy_data_path"]):
        print(f"Creating dummy dataset at: {CONFIG['data']['dummy_data_path']}")
        with open(CONFIG["data"]["dummy_data_path"], 'w') as f:
            json.dump(dummy_data, f, indent=4)

# --- 2. Data Loading and Preprocessing ---
def load_and_preprocess_data(file_path):
    """
    Loads data from the JSON file and preprocesses it into input-summary pairs.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    texts = []
    summaries = []
    for item in data:
        # As per the report, only use valid summarization samples
        if item.get("summarization", {}).get("is_summarization_sample"):
            # Concatenate segmented text to form the input
            input_text = " ".join(item["segmentation"])
            summary_text = item["summarization"]["summary_text"]
            texts.append(input_text)
            summaries.append(summary_text)
    return texts, summaries

class SummarizationDataset(Dataset):
    """
    PyTorch Dataset class for handling the summarization data.
    """
    def __init__(self, texts, summaries, tokenizer, max_input_length, max_target_length):
        self.texts = texts
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        summary = self.summaries[idx]

        # Tokenize the input text
        model_inputs = self.tokenizer(
            text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Tokenize the target summary
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                summary,
                max_length=self.max_target_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids

        # For T5/BART, the decoder input_ids are created by shifting the labels right.
        # The model handles this internally if we just provide 'labels'.
        # We replace padding token id in the labels with -100 to ignore them in loss calculation.
        labels[labels == self.tokenizer.pad_token_id] = -100

        model_inputs["labels"] = labels.squeeze()
        # Squeeze all tensors to remove the batch dimension of 1
        for key in model_inputs:
            model_inputs[key] = model_inputs[key].squeeze()

        return model_inputs

# --- 3. Training and Evaluation Logic ---
def train_one_epoch(model, dataloader, optimizer, lr_scheduler, device, scaler, grad_accum_steps):
    """
    Performs one full epoch of training.
    """
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for i, batch in enumerate(progress_bar):
        batch = {k: v.to(device) for k, v in batch.items()}

        # Mixed precision training
        with torch.cuda.amp.autocast():
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / grad_accum_steps # Scale loss for gradient accumulation

        # Backward pass
        scaler.scale(loss).backward()

        if (i + 1) % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps # Unscale loss for logging
        progress_bar.set_postfix({"loss": total_loss / (i + 1)})

    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, tokenizer, device):
    """
    Evaluates the model on the validation set and computes ROUGE scores.
    """
    model.eval()
    rouge_metric = evaluate.load("rouge")
    all_predictions = []
    all_references = []
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            batch_for_loss = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast():
                outputs = model(**batch_for_loss)
                loss = outputs.loss
            total_loss += loss.item()

            # Generate summaries for ROUGE calculation
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=CONFIG["data"]["max_target_length"],
                num_beams=4,
                early_stopping=True
            )

            # Decode predictions and labels
            predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Replace -100 with pad_token_id to decode reference summaries
            labels = batch["labels"].clone()
            labels[labels == -100] = tokenizer.pad_token_id
            references = tokenizer.batch_decode(labels, skip_special_tokens=True)

            all_predictions.extend(predictions)
            all_references.extend(references)

    # Compute ROUGE scores
    rouge_results = rouge_metric.compute(predictions=all_predictions, references=all_references)
    avg_loss = total_loss / len(dataloader)
    return avg_loss, rouge_results

# --- 4. Main Orchestration ---
def main():
    """
    Main function to run the entire project workflow.
    """
    # Create dummy data and output directory
    create_dummy_dataset()
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # Load and split data (in a real scenario, you'd have a train/val/test split)
    texts, summaries = load_and_preprocess_data(CONFIG["data"]["dummy_data_path"])
    # For this demo, we'll use the same small dataset for train and val
    train_texts, val_texts = texts, texts
    train_summaries, val_summaries = summaries, summaries

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    for model_name, model_config in CONFIG["models"].items():
        print("-" * 50)
        print(f"Processing Model: {model_name.upper()}")
        print("-" * 50)

        # 1. Load Model and Tokenizer
        print("Loading model and tokenizer...")
        model = model_config["model_class"].from_pretrained(model_config["pretrained_name"]).to(device)
        tokenizer = model_config["tokenizer_class"].from_pretrained(model_config["pretrained_name"])

        # 2. Prepare Datasets and DataLoaders
        print("Preparing datasets...")
        train_dataset = SummarizationDataset(
            train_texts, train_summaries, tokenizer,
            CONFIG["data"]["max_input_length"], CONFIG["data"]["max_target_length"]
        )
        val_dataset = SummarizationDataset(
            val_texts, val_summaries, tokenizer,
            CONFIG["data"]["max_input_length"], CONFIG["data"]["max_target_length"]
        )
        train_dataloader = DataLoader(train_dataset, batch_size=CONFIG["training"]["train_batch_size"], shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=CONFIG["training"]["eval_batch_size"])

        # 3. Setup Optimizer and Scheduler
        optimizer = AdamW(model.parameters(), lr=CONFIG["training"]["learning_rate"], weight_decay=CONFIG["training"]["weight_decay"])
        num_training_steps = (len(train_dataloader) * CONFIG["training"]["num_epochs"]) // CONFIG["training"]["gradient_accumulation_steps"]
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )
        scaler = torch.cuda.amp.GradScaler() # For mixed precision

        # 4. Training Loop
        best_val_loss = float("inf")
        epochs_no_improve = 0
        model_save_path = os.path.join(CONFIG["output_dir"], f"best_{model_name}_model.pt")

        start_time = time.time()
        for epoch in range(CONFIG["training"]["num_epochs"]):
            print(f"\n--- Epoch {epoch + 1}/{CONFIG['training']['num_epochs']} ---")
            train_loss = train_one_epoch(model, train_dataloader, optimizer, lr_scheduler, device, scaler, CONFIG["training"]["gradient_accumulation_steps"])
            val_loss, rouge_scores = evaluate_model(model, val_dataloader, tokenizer, device)

            print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"ROUGE Scores: {rouge_scores}")

            # Early stopping and model saving logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                print(f"Validation loss improved. Saving model to {model_save_path}")
                torch.save(model.state_dict(), model_save_path)
            else:
                epochs_no_improve += 1
                print(f"Validation loss did not improve. Early stopping counter: {epochs_no_improve}/{CONFIG['training']['patience']}")

            if epochs_no_improve >= CONFIG["training"]["patience"]:
                print("Early stopping triggered.")
                break
        
        total_time = time.time() - start_time
        print(f"\nTraining for {model_name.upper()} finished in {datetime.timedelta(seconds=int(total_time))}")

        # 5. Final Evaluation and Inference Example
        print("\n--- Final Evaluation on 'Test' Set ---")
        # Load the best model for final evaluation
        model.load_state_dict(torch.load(model_save_path))
        test_loss, test_rouge_scores = evaluate_model(model, val_dataloader, tokenizer, device) # Using val_dataloader as test set
        print(f"Final Test Loss: {test_loss:.4f}")
        print(f"Final ROUGE Scores for {model_name.upper()}:")
        for key, value in test_rouge_scores.items():
            print(f"  {key}: {value*100:.2f}")

        # 6. Inference Example
        print("\n--- Inference Example ---")
        example_text = "Transformers are a type of neural network architecture that have gained popularity. They were introduced in 2017. They are based on the self-attention mechanism and are highly parallelizable, which makes them efficient for training on large amounts of data. They have achieved state-of-the-art results in many NLP tasks."
        summarize(model, tokenizer, device, example_text)


# --- 5. Inference Function ---
def summarize(model, tokenizer, device, text):
    """
    Generates a summary for a given piece of text using a fine-tuned model.
    """
    model.eval()
    inputs = tokenizer(
        text,
        max_length=CONFIG["data"]["max_input_length"],
        truncation=True,
        return_tensors="pt"
    )
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=CONFIG["data"]["max_target_length"],
            num_beams=5,
            early_stopping=True
        )
    
    summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    print(f"\nOriginal Text:\n{text}")
    print(f"\nGenerated Summary:\n{summary}")
    return summary


if __name__ == "__main__":
    main()
