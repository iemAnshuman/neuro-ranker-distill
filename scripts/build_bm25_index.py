import argparse
import os
import subprocess
import json
import tempfile
from pathlib import Path

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Builds a Pyserini BM25 index.")
parser.add_argument(
    "--collection",
    required=True,
    help="Path to the passages.jsonl file with 'pid' and 'text' fields."
)
parser.add_argument(
    "--index_dir",
    required=True,
    help="Directory to store the Lucene index."
)
args = parser.parse_args()

collection_path = Path(args.collection)
index_path = Path(args.index_dir)

# --- Pre-processing Step ---
# Pyserini's default indexer expects a specific JSON format with "id" and "contents" keys.
# Our data has "pid" and "text". We'll create a temporary, correctly-formatted
# collection file for the indexer to use.
print("Preparing data for Pyserini...")
with tempfile.TemporaryDirectory() as temp_dir:
    temp_collection_path = Path(temp_dir) / "collection.jsonl"
    doc_count = 0
    with open(collection_path, 'r') as infile, open(temp_collection_path, 'w') as outfile:
        for line in infile:
            try:
                original_doc = json.loads(line)
                # Remap keys from {"pid": ..., "text": ...} to {"id": ..., "contents": ...}
                new_doc = {
                    "id": original_doc.get("pid"),
                    "contents": original_doc.get("text")
                }
                if new_doc["id"] is None or new_doc["contents"] is None:
                    continue # Skip malformed lines

                outfile.write(json.dumps(new_doc) + '\n')
                doc_count += 1
            except json.JSONDecodeError:
                print(f"Skipping malformed JSON line: {line.strip()}")
                continue

    print(f"Successfully prepared {doc_count} documents.")
    print("Starting indexing process with Pyserini...")

    # --- Indexing Step ---
    # We use subprocess to call Pyserini's command-line indexing tool.
    # This is the standard and most reliable way to create an index.
    command = [
        "python", "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", temp_dir,
        "--index", str(index_path),
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "1",
        "--storePositions", "--storeDocvectors", "--storeRaw"
    ]

    try:
        # Execute the command
        subprocess.run(command, check=True)
        print(f"\nSuccessfully indexed {doc_count} docs to {args.index_dir}")
    except FileNotFoundError:
        print("\nError: 'python' command not found.")
        print("Please ensure Python is in your system's PATH.")
    except subprocess.CalledProcessError as e:
        print(f"\nAn error occurred during indexing: {e}")
        print("Pyserini indexing failed. Please check its logs above for details.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

# The temporary directory and its contents are automatically deleted when the 'with' block exits.