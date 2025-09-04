import argparse
import torch
from torch.multiprocessing import Pool, set_start_method
from functools import partial
from src.neuro_ranker.trainer_teacher import TeacherTrainer, Pair
from src.neuro_ranker.datasets import MSMini
from src.neuro_ranker.utils import set_seed
from tqdm import tqdm

# --- Worker function for parallel processing ---
def create_pair(qrel, qs_dict, ps_dict):
    qid, pid = qrel
    q_text = qs_dict.get(qid)
    p_text = ps_dict.get(pid)
    if q_text and p_text:
        return Pair(q_text, p_text, 1.0)
    return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--model', default='microsoft/MiniLM-L12-H34-uncased')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--max_len', type=int, default=256)
    p.add_argument('--out_dir', required=True)
    args = p.parse_args()
    set_seed(42)

    print("Loading MSMini dataset...")
    mini = MSMini(args.data_dir)
    print("Dataset loaded. Creating pairs in parallel...")

    qs_dict = dict(mini.qs)
    qrels_list = list(mini.qrels)
    worker_func = partial(create_pair, qs_dict=qs_dict, ps_dict=mini.ps)
    num_processes = torch.multiprocessing.cpu_count()
    print(f"Using {num_processes} processes for data preparation...")

    with Pool(num_processes) as pool:
        results = list(tqdm(pool.imap(worker_func, qrels_list, chunksize=1000), total=len(qrels_list)))

    pairs = [p for p in results if p is not None]

    print(f"{len(pairs)} pairs created. Initializing trainer...")
    trainer = TeacherTrainer(args.model, args.lr, args.max_len)
    
    print("Trainer initialized. Starting training...")
    best = trainer.fit(pairs, args.out_dir, epochs=args.epochs, batch=args.batch)
    print('saved', best)

if __name__ == '__main__':
    set_start_method('spawn', force=True)
    main()