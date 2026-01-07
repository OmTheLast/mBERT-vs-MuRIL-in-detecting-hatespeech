import torch
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

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
            console.print(f"[green]✔ {name} Loaded successfully on {self.device}[/green]")
        except Exception as e:
            console.print(f"[red]✘ Failed to load {name}: {e}[/red]")
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
            "label": "TOXIC" if pred_idx == 1 else "Safe",
            "score": confidence,
            "latency": latency_ms
        }

def generate_report(text, res_a, res_b):
    table = Table(title="⚔️ Model Comparison Report ⚔️", box=box.ROUNDED)

    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column(f"mBERT (General)", style="magenta")
    table.add_column(f"MuRIL (Specialist)", style="orange1")
    table.add_column("Winner", style="green")

    # Prediction Row
    winner_pred = "Tie" if res_a['label'] == res_b['label'] else "Disagreement"
    color_a = "red" if res_a['label'] == "TOXIC" else "green"
    color_b = "red" if res_b['label'] == "TOXIC" else "green"
    
    table.add_row(
        "Prediction", 
        f"[{color_a}]{res_a['label']}[/{color_a}]", 
        f"[{color_b}]{res_b['label']}[/{color_b}]", 
        winner_pred
    )

    # Confidence Row
    conf_diff = abs(res_a['score'] - res_b['score'])
    winner_conf = "mBERT" if res_a['score'] > res_b['score'] else "MuRIL"
    table.add_row(
        "Confidence", 
        f"{res_a['score']:.2%}", 
        f"{res_b['score']:.2%}", 
        f"{winner_conf} (+{conf_diff:.1%})"
    )

    # Latency Row
    winner_lat = "mBERT" if res_a['latency'] < res_b['latency'] else "MuRIL"
    table.add_row(
        "Inference Time", 
        f"{res_a['latency']:.1f} ms", 
        f"{res_b['latency']:.1f} ms", 
        f"{winner_lat}" # Faster is better
    )

    console.print(Panel(f"[bold white]Input Text:[/bold white] [italic]{text}[/italic]", border_style="blue"))
    console.print(table)
    print("\n")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    console.rule("[bold blue]Hinglish Toxicity Harness[/bold blue]")
    
    # 1. Load Models (Make sure paths match your folder structure)
    mbert = ModelWrapper("mBERT", "./models/mbert_model")
    muril = ModelWrapper("MuRIL", "./models/muril_model")
    
    console.print("\n[bold]System Ready. Type 'exit' to quit.[/bold]\n")

    while True:
        text = input(">> Enter text to compare: ")
        if text.lower() in ["exit", "quit"]:
            break
        if not text.strip():
            continue
            
        # Run Comparison
        result_mbert = mbert.predict(text)
        result_muril = muril.predict(text)
        
        generate_report(text, result_mbert, result_muril)