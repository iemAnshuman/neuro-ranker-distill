import argparse
import os  # <-- ADDED
from src.neuro_ranker.trainer_teacher import TeacherTrainer, Pair
from src.neuro_ranker.datasets import MSMini
from src.neuro_ranker.utils import set_seed
from tqdm import tqdm


# --- START OF MODIFICATION ---
# Renamed 'main' to 'run_training' and made it accept 'args'
def run_training(args):
    # The 'args' object is now passed in from manage.py
    # We no longer need to parse arguments here.
    # --- END OF MODIFICATION ---

    # --- START OF MODIFICATION 2 ---
    # Check for existing model (we moved this here)
    best_model_path = os.path.join(args.out_dir, "best.pt")
    if os.path.exists(best_model_path):
        print(f"Found existing trained model: {best_model_path}")
        response = input("Do you want to retrain? (This will overwrite the existing model) [y/N]: ")
        if response.lower() != 'y':
            print("Skipping training. Exiting.")
            return  # Exit the function
        else:
            print("Proceeding with retraining...")
    # --- END OF MODIFICATION 2 ---

    set_seed(42)

    print(f"Loading MSMini dataset from: {args.data_dir}...")
    mini = MSMini(args.data_dir)
    print("Dataset loaded. Creating pairs sequentially (optimized)...")

    qs_dict = dict(mini.qs)
    pairs = []

    for qid, pid in tqdm(mini.qrels, desc="Creating Pairs"):
        q = qs_dict.get(qid)
        psg = mini.ps.get(pid)
        if q and psg:
            pairs.append(Pair(q, psg, 1.0))

    print(f"{len(pairs)} pairs created. Initializing trainer...")
    trainer = TeacherTrainer(args.model, args.lr, args.max_len)

    print("Trainer initialized. Starting training...")
    best = trainer.fit(pairs, args.out_dir, epochs=args.epochs, batch=args.batch)
    print("saved", best)


# --- START OF MODIFICATION 3 ---
# This block now lets you run this file directly for testing
if __name__ == "__main__":
    # If run as a standalone script, parse arguments here
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_dir",
        default="../drive/MyDrive/ms_marco_project",
        help="Path to the MS MARCO dataset"
    )
    p.add_argument("--model", default="microsoft/MiniLM-L12-H384-uncased")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    
    run_training(args) # Call the training function
# --- END OF MODIFICATION 3 ---