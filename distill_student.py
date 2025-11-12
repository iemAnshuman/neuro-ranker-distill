import argparse, torch, os
import train_teacher  # <-- ADDED
from src.neuro_ranker.datasets import MSMini
from src.neuro_ranker.trainer_student import StudentTrainer, QP
from src.neuro_ranker.utils import set_seed
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


class DistillDS(Dataset):
    def __init__(self, mini_dataset, teacher_model, teacher_tok, max_len):
        self.mini = mini_dataset
        self.qs = dict(mini_dataset.qs)
        self.ps = mini_dataset.ps
        self.T = teacher_model
        self.T_tok = teacher_tok
        self.max_len = max_len

        print("Validating query-passage pairs...")
        self.qrels = []
        for qid, pid in tqdm(list(mini_dataset.qrels), desc="Filtering pairs"):
            if qid in self.qs and pid in self.ps:
                self.qrels.append((qid, pid))

        print(
            f"Found {len(self.qrels)} valid pairs out of {len(mini_dataset.qrels)} total."
        )

    def __len__(self):
        return len(self.qrels)

    def __getitem__(self, idx):
        qid, pid = self.qrels[idx]
        q_text = self.qs[qid]
        p_text = self.ps[pid]

        enc = self.T_tok(
            q_text,
            p_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        with torch.no_grad():
            device = self.T.device
            inputs = {k: v.to(device) for k, v in enc.items()}
            logit = (
                self.T(**{k: inputs[k] for k in ["input_ids", "attention_mask"]})
                .logits[0, 0]
                .item()
            )

        return {"query": q_text, "passage": p_text, "label": logit}


def run_distillation(args):
    """Main distillation function, called by manage.py"""
    
    # --- START OF MODIFICATION 1 ---
    # Check for the TEACHER model
    if not os.path.exists(args.teacher):
        print(f"Teacher model not found at: {args.teacher}")
        response = input("Do you want to train the teacher model now? [y/N]: ")
        
        if response.lower() == 'y':
            print("--- Starting Teacher Training ---")
            # Create a simple args object for the teacher trainer
            # using the defaults from manage.py
            teacher_args = argparse.Namespace(
                data_dir=args.data_dir,
                out_dir=os.path.dirname(args.teacher), # e.g., .../models/teacher
                model="microsoft/MiniLM-L12-H384-uncased",
                epochs=1,
                lr=2e-5,
                batch=16,
                max_len=256
            )
            train_teacher.run_training(teacher_args)
            print("--- Teacher Training Finished ---")
        else:
            print("Cannot proceed with student training without a teacher model. Exiting.")
            return
    # --- END OF MODIFICATION 1 ---

    # Ensure the student output directory exists
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(42)

    # Load teacher model
    print(f"Loading teacher model from: {args.teacher}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
        
    ckpt = torch.load(args.teacher, map_location=device)
    teacher_name = ckpt.get("name", "microsoft/MiniLM-L12-H384-uncased")
    T_tok = AutoTokenizer.from_pretrained(teacher_name)
    T = AutoModelForSequenceClassification.from_pretrained(teacher_name, num_labels=1)
    T.load_state_dict(ckpt["model"])
    T.to(device)
    T.eval()

    # Create the on-the-fly dataset
    print(f"Setting up dataset from: {args.data_dir}...")
    mini = MSMini(args.data_dir)
    distill_dataset = DistillDS(mini, T, T_tok, args.max_len)

    trainer = StudentTrainer(args.student, args.lr, args.max_len, temperature=args.temp)
    
    # --- START OF MODIFICATION 2 ---
    # Removed the check for student 'best.pt' here.
    # That logic is now inside trainer.fit()
    # --- END OF MODIFICATION 2 ---
    
    best = trainer.fit(
        distill_dataset, args.out_dir, epochs=args.epochs, batch=args.batch
    )
    print("saved", best)


if __name__ == "__main__":
    # This block is for running this script directly
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_dir", 
        default="/content/drive/MyDrive/ms_marco_project", 
        help="Path to the MS MARCO dataset"
    )
    p.add_argument(
        "--teacher", 
        default="/content/drive/MyDrive/ms_marco_project/models/teacher/best.pt",
        help="Path to teacher best.pt"
    )
    p.add_argument("--student", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--temp", type=float, default=3.0)
    p.add_argument(
        "--out_dir", 
        default="/content/drive/MyDrive/ms_marco_project/models/student",
        help="Output directory for student model"
    )
    args = p.parse_args()
    
    run_distillation(args)