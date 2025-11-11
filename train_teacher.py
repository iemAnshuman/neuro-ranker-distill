import argparse
import os
from src.neuro_ranker.trainer_teacher import TeacherTrainer, Pair
from src.neuro_ranker.datasets import MSMini
from src.neuro_ranker.utils import set_seed
from tqdm import tqdm

def run_training(args):
    """Main training function, called by manage.py"""
    best_model_path = os.path.join(args.out_dir, "best.pt")
    if os.path.exists(best_model_path):
        print(f"Found existing trained model: {best_model_path}")
        response = input("Do you want to retrain? (This will overwrite the existing model) [y/N]: ")
        if response.lower() != 'y':
            print("Skipping training. Exiting.")
            return
        else:
            print("Proceeding with retraining...")
    
    # Ensure the output directory exists
    os.makedirs(args.out_dir, exist_ok=True)

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


if __name__ == "__main__":
    # This block is for running this script directly
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_dir",
        default="/content/drive/MyDrive/ms_marco_project",
        help="Path to the MS MARCO dataset"
    )
    p.add_argument("--model", default="microsoft/MiniLM-L12-H384-uncased")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max_len", type=int, default=256)
    # --- START OF MODIFICATION ---
    p.add_argument(
        "--out_dir", 
        default="/content/drive/MyDrive/ms_marco_project/models/teacher",
        help="Output directory for teacher model"
    )
    # --- END OF MODIFICATION ---
    args = p.parse_args()
    
    run_training(args)