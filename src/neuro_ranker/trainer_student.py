from dataclasses import dataclass
from typing import List
import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from .losses import kd_ce # Import actual loss function

@dataclass
class QP:
	query: str
	passage: str
	label: float

class BiEncoder(torch.nn.Module):
	def __init__(self, model_name: str):
		super().__init__()
		self.encoder = AutoModel.from_pretrained(model_name)

	def pool(self, last_hidden_state, attention_mask):
		last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
		return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

	def encode(self, input_ids, attention_mask):
		outputs = self.encoder(input_ids, attention_mask=attention_mask)
		return self.pool(outputs.last_hidden_state, attention_mask)

	def forward(self, q_input_ids, q_attn, p_input_ids, p_attn):
		q_vec = self.encode(q_input_ids, q_attn)
		p_vec = self.encode(p_input_ids, p_attn)
		
		# Normalize for cosine similarity, which is standard for sentence-transformers
		q_vec = torch.nn.functional.normalize(q_vec, p=2, dim=1)
		p_vec = torch.nn.functional.normalize(p_vec, p=2, dim=1)
		
		score = (q_vec * p_vec).sum(dim=1)
		return score, q_vec, p_vec


class QPDS(Dataset):
	def __init__(self, items: List[QP], qtok, ptok, max_len: int):
		self.items = items
		self.qtok = qtok
		self.ptok = ptok
		self.max_len = max_len

	def __len__(self):
		return len(self.items)

	def __getitem__(self, idx):
		item = self.items[idx]
		q_enc = self.qtok(item.query, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
		p_enc = self.ptok(item.passage, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
		return {
			'q_input_ids': q_enc['input_ids'].squeeze(0),
			'q_attn': q_enc['attention_mask'].squeeze(0),
			'p_input_ids': p_enc['input_ids'].squeeze(0),
			'p_attn': p_enc['attention_mask'].squeeze(0),
			'tlogit': torch.tensor(item.label, dtype=torch.float)
		}

# Removed the placeholder kd_ce function as it's now imported

class StudentTrainer:
	def __init__(self, model_name: str, lr: float, max_len: int, temperature: float = 3.0):
		self.qtok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
		self.ptok = self.qtok
		self.model = BiEncoder(model_name)
		self.lr = lr
		self.max_len = max_len
		self.T = temperature

	def fit(self, items: List[QP], out_dir: str, epochs: int = 1, batch: int = 64, ips=False, save_every: int = 1000):
		os.makedirs(out_dir, exist_ok=True)
		ds = QPDS(items, self.qtok, self.ptok, self.max_len)
		dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0)

		opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
		
		start_epoch = 0
		global_step = 0
		best_loss = 1e9

		# --- RESUME LOGIC ---
		ckpt_path = os.path.join(out_dir, 'latest_checkpoint.pt')
		if os.path.exists(ckpt_path):
			print(f"Resuming student training from checkpoint: {ckpt_path}")
			checkpoint = torch.load(ckpt_path)
			self.model.load_state_dict(checkpoint['model_state_dict'])
			opt.load_state_dict(checkpoint['optimizer_state_dict'])
			start_epoch = checkpoint['epoch']
			global_step = checkpoint['global_step']
			best_loss = checkpoint.get('best_loss', 1e9)
			print(f"Resuming from Epoch {start_epoch}, Step {global_step}")
		# --------------------

		self.model.train()
		for ep in range(start_epoch, epochs):
			run = 0.0
			# Correctly handle progress bar for resumed epochs
			prog_bar = tqdm(dl, desc=f"student ep{ep+1}", initial=global_step % len(dl), total=len(dl))
			for i, batch_x in enumerate(prog_bar):
				# Skip steps already completed in a resumed epoch
				if global_step > 0 and i < (global_step % len(dl)):
					continue

				if torch.cuda.is_available():
					for k in batch_x:
						batch_x[k] = batch_x[k].to(self.model.encoder.device)

				s_scores, q, p = self.model(batch_x['q_input_ids'], batch_x['q_attn'], batch_x['p_input_ids'], batch_x['p_attn'])
				st_logits = torch.stack([-s_scores, s_scores], dim=1)
				tch_logits = torch.stack([-batch_x['tlogit'], batch_x['tlogit']], dim=1)
				
				loss_kd = kd_ce(st_logits, tch_logits, self.T)
				loss = loss_kd
				opt.zero_grad()
				loss.backward()
				opt.step()
				run += float(loss.item())
				global_step += 1
				
				# --- PERIODIC CHECKPOINTING ---
				if global_step % save_every == 0:
					torch.save({
						'epoch': ep,
						'global_step': global_step,
						'model_state_dict': self.model.state_dict(),
						'optimizer_state_dict': opt.state_dict(),
						'loss': loss.item(),
						'best_loss': best_loss
					}, ckpt_path)
				# ------------------------------

			avg = run / max(1, len(dl))
			print(f"Epoch {ep+1} avg loss: {avg:.4f}")
			if avg < best_loss:
				best_loss = avg
				best_path = os.path.join(out_dir, 'best.pt')
				torch.save({'model': self.model.state_dict()}, best_path)

		final_path = os.path.join(out_dir, 'best.pt')
		return final_path if os.path.exists(final_path) else "No model saved."