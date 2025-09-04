import argparse, torch
from src.neuro_ranker.datasets import MSMini
from src.neuro_ranker.trainer_student import StudentTrainer, QP
from src.neuro_ranker.utils import set_seed
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# --- UPDATED DATASET CLASS ---
# This class now pre-filters pairs to prevent KeyErrors during training.
class DistillDS(Dataset):
    def __init__(self, mini_dataset, teacher_model, teacher_tok, max_len):
        self.mini = mini_dataset
        self.qs = dict(mini_dataset.qs)
        self.ps = mini_dataset.ps
        self.T = teacher_model
        self.T_tok = teacher_tok
        self.max_len = max_len

        # --- FIX ---
        # Pre-filter qrels to ensure all qids and pids have corresponding text entries.
        # This prevents the DataLoader from crashing on missing keys.
        print("Validating query-passage pairs...")
        self.qrels = []
        for qid, pid in tqdm(list(mini_dataset.qrels), desc="Filtering pairs"):
            if qid in self.qs and pid in self.ps:
                self.qrels.append((qid, pid))
        
        print(f"Found {len(self.qrels)} valid pairs out of {len(mini_dataset.qrels)} total.")
        # --- END FIX ---

    def __len__(self):
        return len(self.qrels)

    def __getitem__(self, idx):
        qid, pid = self.qrels[idx]
        q_text = self.qs[qid]
        p_text = self.ps[pid]
        
        # Teacher scoring is now done here, for one item at a time
        enc = self.T_tok(q_text, p_text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        with torch.no_grad():
            # Move tensors to the same device as the model
            device = self.T.device
            inputs = {k: v.to(device) for k, v in enc.items()}
            logit = self.T(**{k:inputs[k] for k in ['input_ids','attention_mask']}).logits[0,0].item()
            
        return QP(query=q_text, passage=p_text, label=logit)

# --- MAIN SCRIPT (MODIFIED) ---
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--teacher', required=True)
    p.add_argument('--student', default='sentence-transformers/all-MiniLM-L6-v2')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--lr', type=float, default=3e-5)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--max_len', type=int, default=256)
    p.add_argument('--temp', type=float, default=3.0)
    p.add_argument('--out_dir', required=True)
    args = p.parse_args()
    set_seed(42)

    # Load teacher model
    print("Loading teacher model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.teacher, map_location=device)
    teacher_name = ckpt.get('name','microsoft/MiniLM-L12-H384-uncased')
    T_tok = AutoTokenizer.from_pretrained(teacher_name)
    T = AutoModelForSequenceClassification.from_pretrained(teacher_name, num_labels=1)
    T.load_state_dict(ckpt['model'])
    T.to(device) # Move teacher to GPU
    T.eval()

    # Create the on-the-fly dataset
    print("Setting up dataset...")
    mini = MSMini(args.data_dir)
    distill_dataset = DistillDS(mini, T, T_tok, args.max_len)
    
    # Pass the dataset directly to the trainer
    trainer = StudentTrainer(args.student, args.lr, args.max_len, temperature=args.temp)
    # The 'items' argument in fit() now expects a Dataset object, not a list
    best = trainer.fit(distill_dataset, args.out_dir, epochs=args.epochs, batch=args.batch)
    print('saved', best)

if __name__ == '__main__':
    main()