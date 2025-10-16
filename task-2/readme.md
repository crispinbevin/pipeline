# Document Chunking & Dataset Conversion Pipeline

This project processes pre-processed PDF text and extracted tables into semantic chunks suitable for NLP model training or embedding generation.

---

## 📂 Project Structure
```
project/
│
├─ chunk_convert_dataset.py # Main script
├─ dataset/ # Output folder (auto-created)
├─ ../pre-processed-docs.jsonl # Text pages (input)
└─ ../tables/ # Folder with table CSV files
```

---

## ⚙️ Setup Instructions

1. Clone or copy the repository
   ```powershell
   git clone <your-repo-url>
   cd project

    Create and activate a virtual environment (optional but recommended)

    python -m venv venv
    venv\Scripts\activate

    Install dependencies

    pip install nltk tqdm pandas

2. Download NLTK sentence tokenizer
    ```powershell
    python -m nltk.downloader punkt
    ```
    Verify input files

        Ensure pre-processed-docs.jsonl exists one directory above the script.

        Ensure tables/ folder (also one directory above) contains .csv files of extracted tables.

3. Run the Pipeline
```powershell

    python .\chunk_convert_dataset.py