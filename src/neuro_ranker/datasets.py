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
        with open(os.path.join(data_dir, "queries.jsonl")) as f:
            for l in f:
                o = json.loads(l)
                self.qs.append((o["qid"], o["text"]))
        with open(os.path.join(data_dir, "passages.jsonl")) as f:
            for l in f:
                o = json.loads(l)
                self.ps[o["pid"]] = o["text"]
        self.qrels = set()
        with open(os.path.join(data_dir, "qrels.tsv")) as f:
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
