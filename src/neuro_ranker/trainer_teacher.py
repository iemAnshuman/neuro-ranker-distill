from dataclasses import dataclass
from typing import List
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm
from .utils import set_seed

@dataclass
class Pair:
	q: str
	p: str
	y: float

class PairDS(Dataset):
	def __init__(self, pairs: List[Pair], tok, max_len):
		self.pairs = pairs
		self.tok = tok
		self.max_len = max_len

	def __len__(self):
		return len(self.pairs)

	def __getitem__(self, i):
		ex = self.pairs[i]
		enc = self.tok(
			ex.q, ex.p, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt'
		)
		enc = {k: v.squeeze(0) for k, v in enc.items()}
		enc['labels'] = torch.tensor(ex.y, dtype=torch.float)
		return enc

class TeacherTrainer:
	def __init__(self, model_name: str, lr: float, max_len: int):
		self.tok = AutoTokenizer.from_pretrained(model_name)
		self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
		self.lr = lr
		self.max_len = max_len

	def fit(self, pairs: List[Pair], out_dir: str, epochs: int = 1, batch: int = 16):
		ds = PairDS(pairs, self.tok, self.max_len)
		dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0)

		opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
		total = len(dl) * epochs
		sch = get_linear_schedule_with_warmup(opt, num_warmup_steps=int(0.1 * total), num_training_steps=total)
		self.model.train()
		best = None
		best_loss = 1e9
		for ep in range(epochs):
			prog = tqdm(dl, desc=f"teacher ep{ep + 1}")
			run = 0.0
			for batch_x in prog:
				for k in ['input_ids', 'token_type_ids', 'attention_mask', 'labels']:
					if k in batch_x:
						batch_x[k] = batch_x[k].to(self.model.device)
				out = self.model(**{k: batch_x[k] for k in ['input_ids', 'attention_mask', 'labels']})
				loss = out.loss
				opt.zero_grad()
				loss.backward()
				opt.step()
				sch.step()
				run += loss.item()
				prog.set_postfix(loss=f"{loss.item():.4f}")
			avg = run / max(1, len(dl))
			if avg < best_loss:
				best_loss = avg
				os.makedirs(out_dir, exist_ok=True)
				best = os.path.join(out_dir, 'best.pt')
				torch.save({'model': self.model.state_dict(), 'name': self.model.name_or_path}, best)
		return best