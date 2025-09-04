import argparse
from src.neuro_ranker.trainer_teacher import TeacherTrainer, Pair
from src.neuro_ranker.datasets import MSMini
from src.neuro_ranker.utils import set_seed
from tqdm import tqdm

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--model', default='microsoft/MiniLM-L12-H384-uncased')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--max_len', type=int, default=256)
    p.add_argument('--out_dir', required=True)
    args = p.parse_args()
    set_seed(42)

    print("Loading MSMini dataset...")
    mini = MSMini(args.data_dir)
    print("Dataset loaded. Creating pairs sequentially (optimized)...")

    # --- Start of Simple Optimization ---
    # Create the lookup dictionary ONCE. This is the key optimization.
    qs_dict = dict(mini.qs)
    pairs = []
    
    # Use tqdm for a progress bar
    for qid, pid in tqdm(mini.qrels, desc="Creating Pairs"):
        q = qs_dict.get(qid)
        psg = mini.ps.get(pid)
        if q and psg:
            pairs.append(Pair(q, psg, 1.0))
    # --- End of Simple Optimization ---

    print(f"{len(pairs)} pairs created. Initializing trainer...")
    trainer = TeacherTrainer(args.model, args.lr, args.max_len)
    
    print("Trainer initialized. Starting training...")
    best = trainer.fit(pairs, args.out_dir, epochs=args.epochs, batch=args.batch)
    print('saved', best)

if __name__ == '__main__':
    main()