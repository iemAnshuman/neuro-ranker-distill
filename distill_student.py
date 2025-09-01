import argparse, json, os, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.neuro_ranker.datasets import MSMini
from src.neuro_ranker.trainer_student import StudentTrainer, QP
from src.neuro_ranker.utils import set_seed

p = argparse.ArgumentParser()
p.add_argument('--data_dir', required=True)
p.add_argument('--teacher', required=True)
p.add_argument('--student', default='sentence-transformers/all-MiniLM-L6-v2')
p.add_argument('--epochs', type=int, default=1)
p.add_argument('--lr', type=float, default=3e-5)
p.add_argument('--batch', type=int, default=64)
p.add_argument('--max_len', type=int, default=256)
p.add_argument('--temp', type=float, default=3.0)
p.add_argument('--in_batch_negs', type=int, default=1)
p.add_argument('--hard_negs', type=int, default=0)
p.add_argument('--out_dir', required=True)
args = p.parse_args()
set_seed(42)

# Load tiny dataset and create teacher scores for (q,pos)
mini = MSMini(args.data_dir)
# Teacher model
ckpt = torch.load(args.teacher, map_location='cpu')
teacher_name = ckpt.get('name','microsoft/MiniLM-L12-H384-uncased')
from transformers import AutoTokenizer, AutoModelForSequenceClassification
T_tok = AutoTokenizer.from_pretrained(teacher_name)
T = AutoModelForSequenceClassification.from_pretrained(teacher_name, num_labels=1)
T.load_state_dict(ckpt['model'])
T.eval()

items = []
for (qid, pid) in mini.qrels:
    q = dict(mini.qs)[qid]
    psg = mini.ps[pid]
    enc = T_tok(q, psg, truncation=True, padding=True, max_length=args.max_len, return_tensors='pt')
    with torch.no_grad():
        logit = T(**{k:enc[k] for k in ['input_ids','attention_mask']}).logits[0,0].item()
    items.append(QP(q, psg, logit))

trainer = StudentTrainer(args.student, args.lr, args.max_len, temperature=args.temp)
best = trainer.fit(items, args.out_dir, epochs=args.epochs, batch=args.batch)
print('saved', best)