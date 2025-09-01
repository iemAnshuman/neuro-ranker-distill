import json
from typing import List, Tuple
from pyserini.search.lucene import LuceneSearcher

class BM25:
    def __init__(self, index_dir: str, k1: float = 0.9, b: float = 0.4):
        self.searcher = LuceneSearcher(index_dir)
        self.searcher.set_bm25(k1, b)

    def topk(self, query: str, k: int = 100) -> List[Tuple[str, float, str]]:
        hits = self.searcher.search(query, k)
        out = []
        for h in hits:
            pid = h.docid
            score = h.score
            doc = self.searcher.doc(h.docid)
            text = json.loads(doc.raw())["contents"] if doc is not None else ""

            out.append((pid, score, text))
        return out