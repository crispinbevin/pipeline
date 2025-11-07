# Semantic Chunking & Dataset Conversion

Convert preprocessed documents and extracted tables into semantic chunks optimized for AI training, embedding, and RAG systems.

## Overview

`chunk_convert_dataset.py` takes the output from `doc_extract_preprocess.py` and creates semantically meaningful chunks by:

- Splitting text by sentences and grouping them within a token limit
- Converting table rows into readable text format
- Combining both sources into a unified dataset
- Outputting in both JSONL and CSV formats

## Prerequisites

- Python 3.8 or higher
- Virtual environment activated with dependencies installed (see main README)
- Preprocessed documents: `pre-processed-docs.jsonl`
- Extracted tables in `output-tables/` or `tables/` directory

## Setup

### 1. Prepare Input Files

Ensure you have:

```
project/
├── pre-processed-docs.jsonl        # From doc_extract_preprocess.py
├── output-tables/ (or tables/)     # CSV files from doc_extract_preprocess.py
├── chunk_convert_dataset.py
└── venv/
```

### 2. Configure (Optional)

Open `chunk_convert_dataset.py` and adjust the configuration section at the top:

```python
TEXT_FILE_PATH = ...              # Path to pre-processed-docs.jsonl
TABLES_DIR = ...                  # Path to tables directory
OUTPUT_DIR = ...                  # Where chunks will be saved
MAX_TOKENS = 300                  # Approximate max words per chunk
PREVIEW_COUNT = 3                 # Number of chunks to preview in output
```

Default settings work for standard folder structure.

## Usage

### Basic Run

```bash
python chunk_convert_dataset.py
```

### Output

The script creates the `chunked-docs/` directory with:

- **`semantic_chunks_combined.jsonl`** - Line-delimited JSON format
  ```json
  {"source_type": "text", "source_file": "file.pdf", "page_number": 1, "chunk_index": 0, "chunk_text": "..."}
  {"source_type": "table", "source_file": "file_table1.csv", "page_number": null, "chunk_index": 0, "chunk_text": "..."}
  ```

- **`semantic_chunks_combined.csv`** - Spreadsheet format
  | source_type | source_file | page_number | chunk_index | chunk_text |
  |---|---|---|---|---|
  | text | file.pdf | 1 | 0 | ... |
  | table | file_table1.csv | null | 0 | ... |

## Output Format

### Text Chunks
Chunks are created by grouping sentences together until they reach approximately `MAX_TOKENS` words. Each chunk includes:
- `source_type`: "text"
- `source_file`: Original filename
- `page_number`: Page from original document
- `chunk_index`: Index within that page/file
- `chunk_text`: The actual text content

### Table Chunks
Each row from a CSV table becomes a chunk. Columns are converted to "key: value" format:
- `source_type`: "table"
- `source_file`: CSV filename
- `page_number`: null
- `chunk_index`: Row number
- `chunk_text`: "column1: value1 | column2: value2 | ..."

## Configuration Options

| Setting | Default | Description |
|---------|---------|-------------|
| `MAX_TOKENS` | 300 | Approximate maximum words per text chunk |
| `PREVIEW_COUNT` | 3 | Number of chunks to display in console preview |
| `TEXT_FILE_PATH` | `../pre-processed-docs.jsonl` | Path to preprocessed text |
| `TABLES_DIR` | `../tables` | Directory containing CSV table files |
| `OUTPUT_DIR` | `../chunked-docs` | Where to save output files |

## Example Workflow

```bash
# 1. Preprocess documents (creates pre-processed-docs.jsonl and tables)
python doc_extract_preprocess.py ./input-docs

# 2. Convert to semantic chunks
python chunk_convert_dataset.py

# 3. Use chunks for training or embedding
# - Load semantic_chunks_combined.jsonl for streaming
# - Load semantic_chunks_combined.csv for analysis
```

## Output Preview

The script automatically shows a preview of the first few chunks:

```
--- Chunk 1 ---
Type: text | File: document.pdf | Index: 0
This is the first semantic chunk containing multiple sentences grouped together...

--- Chunk 2 ---
Type: table | File: document_table1.csv | Index: 0
product_name: Widget A | price: $19.99 | quantity: 150 | category: Electronics
```

## Using the Output

### For LLM Training / Fine-tuning
Use the JSONL format for streaming:
```python
import json
with open('chunked-docs/semantic_chunks_combined.jsonl') as f:
    for line in f:
        chunk = json.loads(line)
        # Train or embed chunk['chunk_text']
```

### For RAG / Vector Databases
Load and embed all chunks:
```python
import json
chunks = []
with open('chunked-docs/semantic_chunks_combined.jsonl') as f:
    chunks = [json.loads(line) for line in f]
# Embed chunks and store in vector DB
```

### For Analysis
Use the CSV format in Excel, pandas, etc.:
```python
import pandas as pd
df = pd.read_csv('chunked-docs/semantic_chunks_combined.csv')
print(df.groupby('source_type').size())
```

## Troubleshooting

**Error: "pre-processed-docs.jsonl not found"**
- Run `doc_extract_preprocess.py` first
- Check `TEXT_FILE_PATH` configuration

**Error: "Tables directory not found"**
- Ensure tables are extracted from `doc_extract_preprocess.py`
- Check `TABLES_DIR` configuration

**Too few or too many chunks**
- Adjust `MAX_TOKENS` (increase for larger chunks, decrease for smaller)
- Re-run the script

**Empty chunks**
- The script automatically removes empty chunks
- Check source files aren't empty

## Error Logging

The script logs any issues to the console output. Check for:
- Failed CSV reads
- Files that couldn't be processed
- Empty chunks that were dropped