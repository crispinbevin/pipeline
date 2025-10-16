import os
import re
import json
import pandas as pd
import fitz  # PyMuPDF
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders.parsers import RapidOCRBlobParser
import logging

# logging functionality
logging.basicConfig(
    filename='pre-processing-errors.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# clean text utility
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = text.replace('\n', ' ').replace('\t', ' ')
    return text.strip()

# chunked PDF text processor
def process_pdf_in_chunks(directory_path, chunk_size=10):
    output_file = "./pre-processed-docs.jsonl"
    all_results = []

    with open(output_file, "w", encoding="utf-8") as f:
        for filename in os.listdir(directory_path):
            if not filename.endswith(".pdf"):
                continue

            file_path = os.path.join(directory_path, filename)
            print(f"Processing (chunked): {filename}")

            try:
                # open directly with fitz for page-level iteration
                doc = fitz.open(file_path)
                total_pages = len(doc)

                # process in chunks
                for start_page in range(0, total_pages, chunk_size):
                    end_page = min(start_page + chunk_size, total_pages)
                    for page_num in range(start_page, end_page):
                        page = doc.load_page(page_num)
                        text = page.get_text("text")
                        clean = clean_text(text)

                        page_data = {
                            "text": clean,
                            "metadata": {
                                "filename": filename,
                                "page_number": page_num + 1,
                                "character_count": len(clean)
                            }
                        }
                        f.write(json.dumps(page_data, ensure_ascii=False) + "\n")
                        all_results.append(page_data)

                
                doc.close()

            except Exception as e:
                logging.error(f"Failed to process {filename}: {str(e)}")
                continue

    return all_results

# table extraction with chunking
def extract_tables_from_pdf(directory_path, output_dir="../output-tables", chunk_size=10):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(directory_path):
        if not filename.endswith(".pdf"):
            continue

        file_path = os.path.join(directory_path, filename)
        print(f"Extracting tables (chunked, PyMuPDF only): {filename}")

        try:
            doc = fitz.open(file_path)
            total_pages = len(doc)
            table_count = 0

            # process in chunks
            for start_page in range(0, total_pages, chunk_size):
                end_page = min(start_page + chunk_size, total_pages)
                
                for page_num in range(start_page, end_page):
                    try:
                        page = doc.load_page(page_num)
                        tables = page.find_tables()

                        if tables.tables:
                            for idx, table in enumerate(tables.tables, start=1):
                                df = pd.DataFrame(table.extract())
                                csv_filename = f"{os.path.splitext(filename)[0]}_page{page_num+1}_table{idx}.csv"
                                csv_path = os.path.join(output_dir, csv_filename)
                                df.to_csv(csv_path, index=False, encoding="utf-8")
                                table_count += 1
                    
                    except Exception as e:
                        logging.error(f"Failed to extract tables from {filename} page {page_num+1}: {str(e)}")
                        continue

            doc.close()

            if table_count == 0:
                print(f"No tables found in {filename}")
            else:
                print(f"Extracted {table_count} tables from {filename}")

        except Exception as e:
            logging.error(f"Failed to open/process {filename}: {str(e)}")
            continue


# example usage
if __name__ == "__main__":
    pdf_directory = "./input-docs"
    results = process_pdf_in_chunks(pdf_directory, chunk_size=10)
    extract_tables_from_pdf(pdf_directory, chunk_size=10)
    print(f"Processed {len(results)} pages in total.")