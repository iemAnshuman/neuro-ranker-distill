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
        # Changed path to look inside the 'queries' subdirectory
        with open(os.path.join(data_dir, "queries", "train.jsonl")) as f:
        # --- END OF MODIFICATION ---
            for l in f:
                o = json.loads(l)
                self.qs.append((o["qid"], o["text"]))
        
        # This one looks correct based on your screenshot
        with open(os.path.join(data_dir, "collection.jsonl")) as f:
            for l in f:
                o = json.loads(l)
                self.ps[o["pid"]] = o["text"]
        
        self.qrels = set()
        # This one also looks correct
        with open(os.path.join(data_dir, "qrels.train.tsv")) as f:
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