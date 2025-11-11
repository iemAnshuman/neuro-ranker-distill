from dataclasses import dataclass
from typing import List
import json
import os


@dataclass
class Triple:
    qid: str
    query: str
    pid: str
    passage: str
    label: float


class MSMini:
    def __init__(self, data_dir: str):
        self.qs = []
        self.ps = {}
        # --- START OF MODIFICATION ---
        # Changed "queries.jsonl" to "queries.train.jsonl"
        with open(os.path.join(data_dir, "queries.train.jsonl")) as f:
        # --- END OF MODIFICATION ---
            for l in f:
                o = json.loads(l)
                self.qs.append((o["qid"], o["text"]))
        # --- START OF MODIFICATION ---
        # Changed "passages.jsonl" to "collection.jsonl"
        with open(os.path.join(data_dir, "collection.jsonl")) as f:
        # --- END OF MODIFICATION ---
            for l in f:
                o = json.loads(l)
                self.ps[o["pid"]] = o["text"]
        self.qrels = set()
        # --- START OF MODIFICATION ---
        # Changed "qrels.tsv" to "qrels.train.tsv"
        with open(os.path.join(data_dir, "qrels.train.tsv")) as f:
        # --- END OF MODIFICATION ---
            for l in f:
                qid, _q0, pid, rel = l.strip().split("\t")
                self.qrels.add((qid, pid))

    def positive_pairs(self) -> List[Triple]:
        out = []
        for qid, pid in self.qrels:
            qtext = dict(self.qs).get(qid)
            ptext = self.ps.get(pid)
            if qtext is not None and ptext is not None:
                out.append(Triple(qid, qtext, pid, ptext, 1.0))
        return out