import torch
import pandas as pd
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# Initialize Rich Console for pretty printing
console = Console()

class ModelWrapper:
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        console.print(f"[yellow]Loading {name}...[/yellow]", end="\r")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device)
            self.model.eval() # Set to evaluation mode (faster, no training)
            console.print(f"[green]SUCCESS: {name} Loaded successfully on {self.device}[/green]")
        except Exception as e:
            console.print(f"[red]ERROR: Failed to load {name}: {e}[/red]")
            exit()

    def predict(self, text):
        # 1. Start Timer
        start_time = time.time()

        # 2. Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)

        # 3. Inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 4. Process Logic
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()

        # 5. Stop Timer
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000

        return {
            "prediction": pred_idx,
            "label": "TOXIC" if pred_idx == 1 else "Safe",
            "confidence": confidence,
            "latency": latency_ms
        }

def calculate_metrics(true_labels, predictions):
    """Calculate accuracy, precision, recall, and F1 score"""
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary', zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def run_benchmark(test_data, mbert_model, muril_model):
    """Run the benchmark test on both models"""
    results = []
    
    console.print(f"\n[bold blue]Running Benchmark Test on {len(test_data)} samples...[/bold blue]\n")
    
    for idx, row in test_data.iterrows():
        text = row['text']
        true_label = row['label']
        
        # Get predictions from both models
        mbert_result = mbert_model.predict(text)
        muril_result = muril_model.predict(text)
        
        # Store results for metrics calculation
        result = {
            'text': text,
            'true_label': true_label,
            'mbert_prediction': mbert_result['prediction'],
            'muril_prediction': muril_result['prediction'],
            'mbert_confidence': mbert_result['confidence'],
            'muril_confidence': muril_result['confidence'],
            'mbert_latency': mbert_result['latency'],
            'muril_latency': muril_result['latency']
        }
        results.append(result)
        
        # Show progress
        if (idx + 1) % 20 == 0 or idx == len(test_data) - 1:
            console.print(f"[green]Processed {idx + 1}/{len(test_data)} samples...[/green]")
    
    return pd.DataFrame(results)

def generate_benchmark_report(results_df):
    """Generate comprehensive benchmark report"""
    # Calculate metrics for both models
    mbert_metrics = calculate_metrics(results_df['true_label'], results_df['mbert_prediction'])
    muril_metrics = calculate_metrics(results_df['true_label'], results_df['muril_prediction'])
    
    # Calculate average latency
    avg_mbert_latency = results_df['mbert_latency'].mean()
    avg_muril_latency = results_df['muril_latency'].mean()
    
    # Create comparison table
    table = Table(title="Benchmark Results Comparison", box=box.ROUNDED)
    
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column(f"mBERT (General)", style="magenta")
    table.add_column(f"MuRIL (Specialist)", style="orange1")
    table.add_column("Winner", style="green")
    
    # Accuracy Row
    winner_acc = "mBERT" if mbert_metrics['accuracy'] > muril_metrics['accuracy'] else "MuRIL"
    if abs(mbert_metrics['accuracy'] - muril_metrics['accuracy']) < 0.001:
        winner_acc = "Tie"
    table.add_row(
        "Accuracy",
        f"{mbert_metrics['accuracy']:.2%}",
        f"{muril_metrics['accuracy']:.2%}",
        winner_acc
    )
    
    # Precision Row
    winner_prec = "mBERT" if mbert_metrics['precision'] > muril_metrics['precision'] else "MuRIL"
    if abs(mbert_metrics['precision'] - muril_metrics['precision']) < 0.001:
        winner_prec = "Tie"
    table.add_row(
        "Precision",
        f"{mbert_metrics['precision']:.2%}",
        f"{muril_metrics['precision']:.2%}",
        winner_prec
    )
    
    # Recall Row
    winner_rec = "mBERT" if mbert_metrics['recall'] > muril_metrics['recall'] else "MuRIL"
    if abs(mbert_metrics['recall'] - muril_metrics['recall']) < 0.001:
        winner_rec = "Tie"
    table.add_row(
        "Recall",
        f"{mbert_metrics['recall']:.2%}",
        f"{muril_metrics['recall']:.2%}",
        winner_rec
    )
    
    # F1 Score Row
    winner_f1 = "mBERT" if mbert_metrics['f1'] > muril_metrics['f1'] else "MuRIL"
    if abs(mbert_metrics['f1'] - muril_metrics['f1']) < 0.001:
        winner_f1 = "Tie"
    table.add_row(
        "F1-Score",
        f"{mbert_metrics['f1']:.2%}",
        f"{muril_metrics['f1']:.2%}",
        winner_f1
    )
    
    # Average Latency Row
    winner_lat = "mBERT" if avg_mbert_latency < avg_muril_latency else "MuRIL"
    latency_diff = abs(avg_mbert_latency - avg_muril_latency)
    table.add_row(
        "Avg Inference Time",
        f"{avg_mbert_latency:.1f} ms",
        f"{avg_muril_latency:.1f} ms",
        f"{winner_lat} (diff: {latency_diff:.1f}ms)"
    )
    
    # Overall Winner
    scores = [mbert_metrics['accuracy'], mbert_metrics['precision'], mbert_metrics['recall'], mbert_metrics['f1']]
    muril_scores = [muril_metrics['accuracy'], muril_metrics['precision'], muril_metrics['recall'], muril_metrics['f1']]
    
    mbert_win_count = sum(1 for i in range(len(scores)) if scores[i] > muril_scores[i])
    muril_win_count = sum(1 for i in range(len(scores)) if muril_scores[i] > scores[i])
    
    overall_winner = "mBERT" if mbert_win_count > muril_win_count else "MuRIL"
    if mbert_win_count == muril_win_count:
        overall_winner = "Tie"
    
    console.print(f"\n[bold yellow]Overall Summary:[/bold yellow] mBERT wins {mbert_win_count} metrics, MuRIL wins {muril_win_count} metrics")
    console.print(f"[bold green]Overall Winner: {overall_winner}[/bold green]")

    console.print(table)

    # Create detailed statistics table
    detail_table = Table(title="Detailed Statistics", box=box.ROUNDED)
    detail_table.add_column("Model", style="cyan")
    detail_table.add_column("TP", style="green")
    detail_table.add_column("TN", style="green")
    detail_table.add_column("FP", style="red")
    detail_table.add_column("FN", style="red")
    detail_table.add_column("Correct", style="green")
    detail_table.add_column("Incorrect", style="red")
    
    # Calculate TP, TN, FP, FN for mBERT
    mbert_tp = ((results_df['true_label'] == 1) & (results_df['mbert_prediction'] == 1)).sum()
    mbert_tn = ((results_df['true_label'] == 0) & (results_df['mbert_prediction'] == 0)).sum()
    mbert_fp = ((results_df['true_label'] == 0) & (results_df['mbert_prediction'] == 1)).sum()
    mbert_fn = ((results_df['true_label'] == 1) & (results_df['mbert_prediction'] == 0)).sum()
    mbert_correct = mbert_tp + mbert_tn
    mbert_incorrect = mbert_fp + mbert_fn
    
    # Calculate TP, TN, FP, FN for MuRIL
    muril_tp = ((results_df['true_label'] == 1) & (results_df['muril_prediction'] == 1)).sum()
    muril_tn = ((results_df['true_label'] == 0) & (results_df['muril_prediction'] == 0)).sum()
    muril_fp = ((results_df['true_label'] == 0) & (results_df['muril_prediction'] == 1)).sum()
    muril_fn = ((results_df['true_label'] == 1) & (results_df['muril_prediction'] == 0)).sum()
    muril_correct = muril_tp + muril_tn
    muril_incorrect = muril_fp + muril_fn
    
    detail_table.add_row(
        "[magenta]mBERT[/magenta]",
        str(mbert_tp), str(mbert_tn), str(mbert_fp), str(mbert_fn),
        str(mbert_correct), str(mbert_incorrect)
    )
    detail_table.add_row(
        "[orange1]MuRIL[/orange1]",
        str(muril_tp), str(muril_tn), str(muril_fp), str(muril_fn),
        str(muril_correct), str(muril_incorrect)
    )
    
    console.print(detail_table)
    
    return {
        'mbert_metrics': mbert_metrics,
        'muril_metrics': muril_metrics,
        'mbert_latency': avg_mbert_latency,
        'muril_latency': avg_muril_latency,
        'overall_winner': overall_winner
    }

def main():
    console.rule("[bold blue]Hinglish Hate Speech Benchmark Test[/bold blue]")
    
    # Load test data
    console.print("[yellow]Loading benchmark test data...[/yellow]")
    test_data = pd.read_csv('benchmark_test.csv')
    console.print(f"[green]Loaded {len(test_data)} test samples[/green]")
    
    # Load Models (Make sure paths match your folder structure)
    mbert = ModelWrapper("mBERT", "./models/mbert_model")
    muril = ModelWrapper("MuRIL", "./models/muril_model")
    
    # Run benchmark
    results_df = run_benchmark(test_data, mbert, muril)
    
    # Generate report
    report = generate_benchmark_report(results_df)
    
    # Save detailed results to CSV
    results_df.to_csv('benchmark_results_detailed.csv', index=False)
    console.print(f"\n[green]SUCCESS: Detailed results saved to 'benchmark_results_detailed.csv'[/green]")
    
    # Print summary
    console.print(f"\n[bold blue]Benchmark Complete![/bold blue]")
    console.print(f"mBERT Accuracy: {report['mbert_metrics']['accuracy']:.2%}")
    console.print(f"MuRIL Accuracy: {report['muril_metrics']['accuracy']:.2%}")
    console.print(f"Overall Winner: {report['overall_winner']}")
    
if __name__ == "__main__":
    main()