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

	def fit(self, pairs: List[Pair], out_dir: str, epochs: int = 1, batch: int = 16, save_every: int = 1000):
		os.makedirs(out_dir, exist_ok=True)
		ds = PairDS(pairs, self.tok, self.max_len)
		dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0)

		opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
		total_steps = len(dl) * epochs
		sch = get_linear_schedule_with_warmup(opt, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
		
		start_epoch = 0
		global_step = 0
		best_loss = 1e9
		
		# --- RESUME LOGIC ---
		ckpt_path = os.path.join(out_dir, 'latest_checkpoint.pt')
		if os.path.exists(ckpt_path):
			print(f"Resuming training from checkpoint: {ckpt_path}")
			checkpoint = torch.load(ckpt_path)
			self.model.load_state_dict(checkpoint['model_state_dict'])
			opt.load_state_dict(checkpoint['optimizer_state_dict'])
			sch.load_state_dict(checkpoint['scheduler_state_dict'])
			start_epoch = checkpoint['epoch']
			global_step = checkpoint['global_step']
			best_loss = checkpoint.get('best_loss', 1e9)
			print(f"Resuming from Epoch {start_epoch}, Step {global_step}")
		# --------------------

		self.model.train()
		for ep in range(start_epoch, epochs):
			prog = tqdm(dl, desc=f"teacher ep{ep + 1}", initial=global_step % len(dl), total=len(dl))
			run = 0.0
			for i, batch_x in enumerate(prog):
				# Skip steps already completed in a resumed epoch
				if global_step > 0 and i < (global_step % len(dl)):
					continue

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
				global_step += 1
				prog.set_postfix(loss=f"{loss.item():.4f}")

				# --- PERIODIC CHECKPOINTING ---
				if global_step % save_every == 0:
					torch.save({
						'epoch': ep,
						'global_step': global_step,
						'model_state_dict': self.model.state_dict(),
						'optimizer_state_dict': opt.state_dict(),
						'scheduler_state_dict': sch.state_dict(),
						'loss': loss.item(),
						'best_loss': best_loss
					}, ckpt_path)
					# print(f"\nSaved checkpoint at step {global_step}")
				# ------------------------------

			avg = run / max(1, len(dl))
			if avg < best_loss:
				best_loss = avg
				best_path = os.path.join(out_dir, 'best.pt')
				torch.save({'model': self.model.state_dict(), 'name': self.model.name_or_path}, best_path)
		
		return os.path.join(out_dir, 'best.pt')