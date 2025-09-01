import argparse
import os
import pandas as pd
from tqdm import tqdm
import json

def main():
    parser = argparse.ArgumentParser(description="Convert MS MARCO TSV data to JSONL format.")
    parser.add_argument(
        '--collection_path',
        required=True,
        help="Path to the collection.tsv file."
    )
    parser.add_argument(
        '--queries_path',
        required=True,
        help="Path to the queries.train.tsv file."
    )
    parser.add_argument(
        '--out_dir',
        required=True,
        help="Directory to save the output JSONL files."
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_passages_path = os.path.join(args.out_dir, 'passages.jsonl')
    out_queries_path = os.path.join(args.out_dir, 'queries.jsonl')

    # --- Process the Passages Collection ---
    print(f"Processing collection file: {args.collection_path}")
    # Use pandas chunking for memory efficiency
    chunk_iter = pd.read_csv(
        args.collection_path,
        sep='\t',
        header=None,
        names=['pid', 'text'],
        chunksize=100000
    )

    with open(out_passages_path, 'w') as f:
        for chunk in tqdm(chunk_iter, desc="Converting passages.tsv to passages.jsonl"):
            for _, row in chunk.iterrows():
                # The JSON format our project expects
                doc = {'pid': str(row['pid']), 'text': row['text']}
                f.write(json.dumps(doc) + '\n')
    print(f"Saved passages to {out_passages_path}")


    # --- Process the Queries ---
    print(f"Processing queries file: {args.queries_path}")
    queries_df = pd.read_csv(
        args.queries_path,
        sep='\t',
        header=None,
        names=['qid', 'text']
    )

    with open(out_queries_path, 'w') as f:
        for _, row in tqdm(queries_df.iterrows(), total=len(queries_df), desc="Converting queries.tsv to queries.jsonl"):
            query = {'qid': str(row['qid']), 'text': row['text']}
            f.write(json.dumps(query) + '\n')
    print(f"Saved queries to {out_queries_path}")

if __name__ == '__main__':
    main()