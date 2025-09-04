import argparse, json, os
from src.neuro_ranker.trainer_teacher import TeacherTrainer, Pair
from src.neuro_ranker.datasets import MSMini
from src.neuro_ranker.utils import set_seed

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
print("Dataset loaded. Creating pairs...")

pairs = []
for (qid, pid) in mini.qrels:
    q = dict(mini.qs)[qid]
    psg = mini.ps[pid]
    pairs.append(Pair(q, psg, 1.0))
print(f"{len(pairs)} pairs created. Initializing trainer...")

trainer = TeacherTrainer(args.model, args.lr, args.max_len)
print("Trainer initialized. Starting training...")
best = trainer.fit(pairs, args.out_dir, epochs=args.epochs, batch=args.batch)
print('saved', best)