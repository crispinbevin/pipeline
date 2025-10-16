import os
import json
import csv
import re
from tqdm import tqdm

import pandas as pd
import nltk
nltk.download("punkt_tab")

from nltk.tokenize import sent_tokenize

# -------- CONFIGURATION --------
TEXT_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../pre-processed-docs.jsonl"))
TABLES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../tables"))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../chunked-docs"))
MAX_TOKENS = 300  # Approximate max words per chunk
PREVIEW_COUNT = 3
# --------------------------------


def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def clean_text(text):
    """Normalize whitespace, remove junk characters."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text, max_tokens=300):
    """Split text into sentence-based semantic chunks."""
    sentences = sent_tokenize(text)
    chunks, current_chunk, word_count = [], [], 0

    for sentence in sentences:
        words = sentence.split()
        if word_count + len(words) > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk, word_count = [], 0
        current_chunk.append(sentence)
        word_count += len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


def process_text_docs():
    """Load and chunk textual data."""
    data = load_jsonl(TEXT_FILE_PATH)
    chunks = []

    print(f"📘 Processing text file: {TEXT_FILE_PATH}")
    for entry in tqdm(data):
        text = clean_text(entry["text"])
        file_name = entry["metadata"].get("file_name", "unknown")
        page_number = entry["metadata"].get("page_number", -1)
        text_chunks = chunk_text(text, max_tokens=MAX_TOKENS)

        for idx, chunk in enumerate(text_chunks):
            chunks.append({
                "source_type": "text",
                "source_file": file_name,
                "page_number": page_number,
                "chunk_index": idx,
                "chunk_text": chunk
            })

    return chunks


def process_table_csvs():
    """Load and process each CSV table row-wise."""
    table_chunks = []
    if not os.path.exists(TABLES_DIR):
        print(f"⚠️ Tables directory not found: {TABLES_DIR}")
        return table_chunks

    csv_files = [f for f in os.listdir(TABLES_DIR) if f.endswith(".csv")]
    if not csv_files:
        print("⚠️ No CSV tables found.")
        return table_chunks

    print(f"📊 Processing tables from: {TABLES_DIR}")
    for csv_file in tqdm(csv_files):
        csv_path = os.path.join(TABLES_DIR, csv_file)
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"❌ Failed to read {csv_file}: {e}")
            continue

        for i, row in df.iterrows():
            row_text = " | ".join([f"{col}: {str(val)}" for col, val in row.items() if pd.notna(val)])
            row_text = clean_text(row_text)
            if not row_text:
                continue

            table_chunks.append({
                "source_type": "table",
                "source_file": csv_file,
                "page_number": None,
                "chunk_index": i,
                "chunk_text": row_text
            })

    return table_chunks


def validate_chunks(chunks):
    """Remove empty or malformed chunks."""
    valid = [c for c in chunks if c.get("chunk_text", "").strip()]
    dropped = len(chunks) - len(valid)
    if dropped:
        print(f"⚠️ Dropped {dropped} empty chunks.")
    return valid


def write_jsonl(chunks, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in chunks:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")


def write_csv(chunks, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(chunks[0].keys()))
        writer.writeheader()
        writer.writerows(chunks)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("🚀 Starting semantic + table chunking pipeline...")
    text_chunks = process_text_docs()
    table_chunks = process_table_csvs()

    all_chunks = text_chunks + table_chunks
    all_chunks = validate_chunks(all_chunks)

    jsonl_path = os.path.join(OUTPUT_DIR, "semantic_chunks_combined.jsonl")
    csv_path = os.path.join(OUTPUT_DIR, "semantic_chunks_combined.csv")

    write_jsonl(all_chunks, jsonl_path)
    write_csv(all_chunks, csv_path)

    print(f"\n✅ Chunking complete. {len(all_chunks)} total chunks saved to:")
    print(f"   JSONL: {jsonl_path}")
    print(f"   CSV:   {csv_path}")

    if PREVIEW_COUNT > 0:
        print("\n🔍 Preview of combined chunks:")
        for i, chunk in enumerate(all_chunks[:PREVIEW_COUNT]):
            print(f"--- Chunk {i+1} ---")
            print(f"Type: {chunk['source_type']} | File: {chunk['source_file']} | Index: {chunk['chunk_index']}")
            print(chunk['chunk_text'][:400], "\n")


if __name__ == "__main__":
    main()
