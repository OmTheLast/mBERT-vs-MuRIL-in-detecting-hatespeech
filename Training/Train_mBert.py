import re
import pandas as pd
import numpy as np
import torch
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset

def main(data_path=None):
    # Clear memory (important for local runs if you have limited VRAM)
    torch.cuda.empty_cache()
    gc.collect()

    # --- 1. Data Loading and Initial Cleaning ---
    # Load the dataset from the parent directory or specified path
    import os
    dataset_path = data_path or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'combined_hate_speech_dataset.csv')
    print(f"Loading dataset from: {dataset_path}")

    # Load the file into Pandas with error handling for different encodings
    try:
        df = pd.read_csv(dataset_path, encoding='utf-8')
    except:
        df = pd.read_csv(dataset_path, encoding='latin-1')

    print(f"Initial DataFrame size: {len(df)}")

    # Keep only the columns we need
    df = df[['text', 'hate_label']]

    # The Cleaning Function
    def clean_hinglish_text(text):
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    print("\nCleaning text...")
    df['clean_text'] = df['text'].apply(clean_hinglish_text)
    print("Text cleaning complete.")

    # --- 2. Data Splitting ---
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        df['clean_text'],
        df['hate_label'],
        test_size=0.2,
        random_state=42
    )

    print(f"\nTraining on: {len(X_train_full)} samples")
    print(f"Testing on:  {len(X_test_full)} samples")

    # --- 3. Tokenization (mBERT) ---
    model_checkpoint = "bert-base-multilingual-cased"
    print(f"\nLoading tokenizer for {model_checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # Convert to Hugging Face Dataset
    train_data_full = pd.DataFrame({'text': X_train_full, 'label': y_train_full})
    test_data_full = pd.DataFrame({'text': X_test_full, 'label': y_test_full})
    hf_train_full = Dataset.from_pandas(train_data_full)
    hf_test_full = Dataset.from_pandas(test_data_full)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    print("Tokenizing full dataset for mBERT...")
    tokenized_train_full = hf_train_full.map(tokenize_function, batched=True)
    tokenized_test_full = hf_test_full.map(tokenize_function, batched=True)
    print("Tokenization complete.")

    # --- 4. Define Metrics Function ---
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        f1 = f1_score(labels, preds, average="weighted")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1}

    # --- 5. Model Loading and Training (mBERT) ---
    print(f"\nLoading {model_checkpoint} model...")
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

    # Training Arguments
    training_args = TrainingArguments(
        output_dir="./mbert_full_results", # Local directory for results
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False, # Important for local script
        report_to="none", # Disable reporting to any external services like wandb
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_full,
        eval_dataset=tokenized_test_full,
        compute_metrics=compute_metrics,
    )

    print("\nStarting mBERT Full Dataset Training... (This will take some time)")
    trainer.train()
    print("Training complete.")

    # --- 6. Save the Trained Model and Tokenizer ---
    save_path = "./Hinglish_Hate_Model_mBert"

    import os
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print(f"\nSaving model to {save_path}...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("Model and tokenizer saved locally.")

    # --- 7. Prediction Function ---
    def predict_hate_speech(text, model, tokenizer):
        # Ensure the model is in evaluation mode
        model.eval()

        # Tokenize the text
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)

        # Move model to the same device as inputs
        model.to(device)

        # Get prediction from the model
        with torch.no_grad():
            outputs = model(**inputs)

        # Convert numbers to probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Get the winner
        pred_label = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_label].item()

        # Print result
        label_map = {0: "Non-Hate (Safe)", 1: "Hate Speech (Toxic)"}
        print(f"Text: '{text}'")
        print(f"Prediction: {label_map[pred_label]} (Confidence: {confidence:.2f})\n")

    # --- TEST ZONE ---
    print("\n--- MBERT MODEL TESTING ---\n")
    predict_hate_speech("You are a great person", model, tokenizer)
    predict_hate_speech("Tu pagal hai kya", model, tokenizer)
    predict_hate_speech("Go back to your country", model, tokenizer)
    predict_hate_speech("Tera dimaag kharaab hai", model, tokenizer)
    predict_hate_speech("I hate you, you are evil.", model, tokenizer)
    predict_hate_speech("This is a lovely morning.", model, tokenizer)

if __name__ == "__main__":
    main()