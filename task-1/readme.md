# Document Processing Scripts Setup

This guide will help you set up and run two document processing scripts: `doc_extract_preprocess.py` and `benchmark.py`.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Setup Instructions

### 1. Create a Virtual Environment

Navigate to your project directory and create a virtual environment:

```bash
python -m venv venv
```

### 2. Activate the Virtual Environment

**On Windows:**
```bash
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
source venv/bin/activate
```

You should see `(venv)` appear in your terminal prompt.

### 3. Install Dependencies

Install all required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Project Structure

Create this folder structure in your project directory (optional, but recommended):

```
project/
├── input-docs/           # Place your PDF and DOCX files here
├── output-tables/        # Extracted tables will be saved here
├── benchmark-results/    # Benchmark results will be saved here
├── doc_extract_preprocess.py
├── benchmark.py
├── requirements.txt
└── venv/
```

### Create the folders:

```bash
mkdir input-docs output-tables benchmark-results
```

## Usage

### Script 1: Document Extraction & Preprocessing

This script extracts text from PDFs and DOCX files, cleans it, and splits it into chunks. It also extracts all tables as CSV files.

**Process a single file:**
```bash
python doc_extract_preprocess.py path/to/your/file.pdf
```

or

```bash
python doc_extract_preprocess.py path/to/your/file.docx
```

**Process a directory:**
```bash
python doc_extract_preprocess.py ./input-docs
```

**Output:**
- `pre-processed-docs.jsonl` - Cleaned text chunks with metadata
- `output-tables/` - Extracted tables as CSV files

### Script 2: PDF Table Extraction Benchmark

This script benchmarks different table extraction libraries (PyMuPDF, Camelot, pdfplumber) to compare performance and accuracy.

**Benchmark a single PDF:**
```bash
python benchmark.py path/to/your/file.pdf
```

**Benchmark a directory:**
```bash
python benchmark.py ./input-docs
```

**Output:**
- `benchmark-results/benchmark_results_[timestamp].csv` - Detailed metrics
- `benchmark-results/benchmark_summary_[timestamp].json` - Summary statistics
- `benchmark-results/extracted_tables/` - Tables extracted by each library (for comparison)

## Example Workflow

1. Place your PDF or DOCX files in the `input-docs/` folder
2. Run the preprocessing script:
   ```bash
   python doc_extract_preprocess.py ./input-docs
   ```
3. Run the benchmark to compare extraction libraries:
   ```bash
   python benchmark.py ./input-docs
   ```
4. Check the output:
   - Preprocessed text: `pre-processed-docs.jsonl`
   - Extracted tables: `output-tables/`
   - Benchmark results: `benchmark-results/`

## Error Handling

Both scripts log errors to:
- `pre-processing-errors.log` - Errors from doc_extract_preprocess.py
- `benchmark-errors.log` - Errors from benchmark.py

Check these files if something goes wrong.

## Deactivating the Virtual Environment

When you're done, deactivate the virtual environment:

```bash
deactivate
```

## Troubleshooting

**Module not found errors:** Make sure you've activated the virtual environment and installed dependencies with `pip install -r requirements.txt`.

**Permission denied (Linux/macOS):** Try running `chmod +x doc_extract_preprocess.py benchmark.py` first.

**Missing dependencies:** Reinstall with `pip install -r requirements.txt --upgrade`.