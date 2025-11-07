```
git clone https://github.com/crispinbevin/pipeline

cd task-1 # or task of choice

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt
```

# Document Processing Pipeline

A comprehensive pipeline for extracting, processing, and converting documents into semantic chunks optimized for AI applications.

## Pipeline Overview

```
PDF/DOCX Files
     ↓
[Extraction & Preprocessing] → Text + Tables
     ↓
[Benchmarking] → Performance Comparison
     ↓
[Semantic Chunking] → AI-Ready Dataset
     ↓
JSONL + CSV Output
```

## Key Stages

### 1. Document Extraction & Preprocessing

**Input:** PDF and DOCX files

**Process:**
- Extracts text from PDFs using PyMuPDF with fallback OCR via RapidOCR
- Extracts paragraphs from DOCX files using python-docx
- Cleans text by removing URLs, normalizing whitespace, and fixing encoding
- Splits documents into manageable chunks based on page/paragraph count
- Extracts all tables and saves as CSV files

**Output:** 
- `pre-processed-docs.jsonl` - Cleaned text chunks with metadata
- `output-tables/` - Extracted tables as CSV files

**Technologies:** PyMuPDF, python-docx, pandas, regex

---

### 2. Table Extraction Benchmarking

**Input:** PDF files

**Process:**
- Compares three table extraction methods:
  - **PyMuPDF** - Built-in table detection
  - **Camelot** - Stream-based table parsing
  - **pdfplumber** - Precise cell extraction
- Measures performance metrics: extraction time, memory usage, accuracy
- Extracts tables using each library for comparison

**Output:**
- Benchmark reports (CSV + JSON) with execution times and memory usage
- Extracted tables organized by library

**Use Case:** Determine the best extraction method for your document types

**Technologies:** PyMuPDF, Camelot, pdfplumber, pandas

---

### 3. Semantic Chunking & Dataset Conversion

**Input:** Preprocessed text + extracted tables

**Process:**
- Tokenizes text into sentences using NLTK
- Groups sentences into semantic chunks (~300 tokens per chunk by default)
- Converts table rows into readable text format ("column: value" pairs)
- Validates and deduplicates chunks
- Combines text and table chunks into unified dataset

**Output:**
- `chunked-docs/semantic_chunks_combined.jsonl` - Streaming format
- `chunked-docs/semantic_chunks_combined.csv` - Analysis format

**Technologies:** NLTK, pandas, JSON/CSV processing

---

## End-to-End Workflow

1. **Extract** documents (PDFs/DOCX) → get raw text and tables
2. **Benchmark** extraction methods → choose optimal library
3. **Chunk** into semantic units → create AI-ready dataset
4. **Output** in multiple formats → use for training, RAG, or embedding

## Use Cases

- **LLM Fine-tuning** - Semantic chunks sized for model context windows
- **Retrieval-Augmented Generation (RAG)** - Chunks ready for vector embedding and retrieval
- **Knowledge Extraction** - Structured data from unstructured documents
- **Document Analysis** - Quantifiable metrics on extraction performance

## Key Features

✓ Multi-format support (PDF, DOCX)  
✓ Comparative benchmarking of extraction methods  
✓ Automatic table detection and extraction  
✓ Semantic sentence-based chunking  
✓ Dual output formats (JSONL for streaming, CSV for analysis)  
✓ Comprehensive error logging  
✓ Scalable directory processing  

## Getting Started

See the individual README files for setup and detailed usage:
- **Setup & Installation** → [Main README](./README.md)
- **Document Extraction** → [doc_extract_preprocess.py](./README_EXTRACT.md)
- **Semantic Chunking** → [chunk_convert_dataset.py](./README_CHUNK.md)