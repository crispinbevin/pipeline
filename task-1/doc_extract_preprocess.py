import os
import json
import sys
import time
import tracemalloc
import pandas as pd
import fitz
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# Try importing optional libraries
try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False

try:
    from pdfminer.high_level import extract_text
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

# Logging setup
logging.basicConfig(
    filename='benchmark-errors.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@dataclass
class ExtractionMetrics:
    library: str
    filename: str
    tables_extracted: int
    total_cells: int
    avg_rows_per_table: float
    avg_cols_per_table: float
    execution_time: float
    memory_used_mb: float
    error: str = None

class PDFTableBenchmark:
    def __init__(self, output_dir="./benchmark-results"):
        # Initializes the benchmark class.
        # Inputs: output_dir (str) - directory where results will be saved.
        # Output: None (sets up instance variables and directories)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results: List[ExtractionMetrics] = []

    def extract_with_pymupdf(self, file_path: str) -> Tuple[int, List[pd.DataFrame], float, float]:
        # Extracts tables from a PDF using PyMuPDF.
        # Inputs: file_path (str) - path to the PDF file.
        # Output: tuple(table_count, tables_list, execution_time, memory_used_mb)
        tracemalloc.start()
        start_time = time.time()
        tables = []
        table_count = 0

        try:
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_tables = page.find_tables()
                
                if page_tables.tables:
                    for table in page_tables.tables:
                        df = pd.DataFrame(table.extract())
                        tables.append(df)
                        table_count += 1
            
            doc.close()
        except Exception as e:
            logging.error(f"PyMuPDF error on {file_path}: {str(e)}")
            raise

        exec_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        memory_mb = peak / 1024 / 1024
        tracemalloc.stop()

        return table_count, tables, exec_time, memory_mb

    def extract_with_camelot(self, file_path: str) -> Tuple[int, List[pd.DataFrame], float, float]:
        # Extracts tables using Camelot library.
        # Inputs: file_path (str) - path to the PDF file.
        # Output: tuple(table_count, tables_list, execution_time, memory_used_mb)
        if not CAMELOT_AVAILABLE:
            raise ImportError("Camelot not installed")
        
        tracemalloc.start()
        start_time = time.time()
        tables = []

        try:
            camelot_tables = camelot.read_pdf(file_path, pages='all', flavor='stream')
            tables = [table.df for table in camelot_tables]
            table_count = len(camelot_tables)
        except Exception as e:
            logging.error(f"Camelot error on {file_path}: {str(e)}")
            raise

        exec_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        memory_mb = peak / 1024 / 1024
        tracemalloc.stop()

        return table_count, tables, exec_time, memory_mb

    def extract_with_pdfplumber(self, file_path: str) -> Tuple[int, List[pd.DataFrame], float, float]:
        # Extracts tables using pdfplumber.
        # Inputs: file_path (str) - path to the PDF file.
        # Output: tuple(table_count, tables_list, execution_time, memory_used_mb)
        if not PDFPLUMBER_AVAILABLE:
            raise ImportError("pdfplumber not installed")
        
        tracemalloc.start()
        start_time = time.time()
        tables = []

        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_tables = page.extract_tables()
                    if page_tables:
                        for table in page_tables:
                            df = pd.DataFrame(table)
                            tables.append(df)
        except Exception as e:
            logging.error(f"pdfplumber error on {file_path}: {str(e)}")
            raise

        exec_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        memory_mb = peak / 1024 / 1024
        tracemalloc.stop()

        return len(tables), tables, exec_time, memory_mb

    def calculate_metrics(self, library: str, filename: str, table_count: int, 
                         tables: List[pd.DataFrame], exec_time: float, memory_mb: float) -> ExtractionMetrics:
        # Calculates and returns aggregated metrics for a given extraction run.
        # Inputs: library name, filename, table_count, tables list, exec_time, memory_mb
        # Output: ExtractionMetrics dataclass instance with aggregated statistics
        total_cells = 0
        total_rows = 0
        total_cols = 0

        for df in tables:
            total_cells += df.shape[0] * df.shape[1]
            total_rows += df.shape[0]
            total_cols += df.shape[1]

        avg_rows = total_rows / table_count if table_count > 0 else 0
        avg_cols = total_cols / table_count if table_count > 0 else 0

        return ExtractionMetrics(
            library=library,
            filename=filename,
            tables_extracted=table_count,
            total_cells=total_cells,
            avg_rows_per_table=round(avg_rows, 2),
            avg_cols_per_table=round(avg_cols, 2),
            execution_time=round(exec_time, 4),
            memory_used_mb=round(memory_mb, 2)
        )

    def save_tables_to_csv(self, tables: List[pd.DataFrame], library: str, filename: str, page_num: int = None):
        # Saves extracted tables to CSV files per library.
        # Inputs: tables list, library name, filename, optional page number.
        # Output: list of saved CSV file paths.
        base_name = os.path.splitext(filename)[0]
        lib_folder = os.path.join(self.output_dir, "extracted_tables", library)
        os.makedirs(lib_folder, exist_ok=True)
        
        saved_files = []
        for idx, table in enumerate(tables, 1):
            if page_num is not None:
                csv_filename = f"{base_name}_{library}_page{page_num}_table{idx}.csv"
            else:
                csv_filename = f"{base_name}_{library}_table{idx}.csv"
            
            csv_path = os.path.join(lib_folder, csv_filename)
            table.to_csv(csv_path, index=False, encoding='utf-8')
            saved_files.append(csv_path)
        
        return saved_files

    def benchmark_file(self, file_path: str):
        # Runs table extraction benchmarking for a single PDF file across all libraries.
        # Inputs: file_path (str) - path to a PDF file.
        # Output: None (updates self.results list with metrics for each library)
        filename = os.path.basename(file_path)
        print(f"\n{'='*60}")
        print(f"Benchmarking: {filename}")
        print(f"{'='*60}")

        libraries = {
            'PyMuPDF': (self.extract_with_pymupdf, True),
            'Camelot': (self.extract_with_camelot, CAMELOT_AVAILABLE),
            'pdfplumber': (self.extract_with_pdfplumber, PDFPLUMBER_AVAILABLE),
        }

        for lib_name, (extract_func, available) in libraries.items():
            if not available:
                print(f"⚠️  {lib_name}: Not installed (skipping)")
                continue

            try:
                print(f"🔄 Processing with {lib_name}...", end=' ')
                table_count, tables, exec_time, memory_mb = extract_func(file_path)
                
                metrics = self.calculate_metrics(lib_name, filename, table_count, tables, exec_time, memory_mb)
                self.results.append(metrics)
                
                saved_files = self.save_tables_to_csv(tables, lib_name, filename)
                print(f"✓ {table_count} tables extracted and saved")
                
                if saved_files:
                    print(f"   └─ Saved to: {os.path.dirname(saved_files[0])}/")
            except Exception as e:
                error_msg = str(e)
                logging.error(f"{lib_name} failed on {filename}: {error_msg}")
                metrics = ExtractionMetrics(
                    library=lib_name,
                    filename=filename,
                    tables_extracted=0,
                    total_cells=0,
                    avg_rows_per_table=0,
                    avg_cols_per_table=0,
                    execution_time=0,
                    memory_used_mb=0,
                    error=error_msg
                )
                self.results.append(metrics)
                print(f"✗ Error: {error_msg[:50]}")

    def benchmark_directory(self, directory_path: str):
        # Runs benchmarking for all PDF files in a directory.
        # Inputs: directory_path (str) - directory containing PDF files.
        # Output: None (calls benchmark_file for each file)
        pdf_files = [f for f in os.listdir(directory_path) if f.endswith('.pdf')]
        
        if not pdf_files:
            print(f"No PDF files found in {directory_path}")
            return

        print(f"Found {len(pdf_files)} PDF file(s)")
        
        for pdf_file in pdf_files:
            self.benchmark_file(os.path.join(directory_path, pdf_file))

    def generate_report(self):
        # Generates and prints summary and detailed benchmark reports.
        # Inputs: None (uses collected self.results data)
        # Outputs: None (prints and saves CSV + JSON summaries)
        if not self.results:
            print("No results to report")
            return

        df_results = pd.DataFrame([asdict(r) for r in self.results])
        
        print(f"\n{'='*80}")
        print("BENCHMARK SUMMARY - Aggregated by Library")
        print(f"{'='*80}\n")
        
        summary = df_results[df_results['error'].isna()].groupby('library').agg({
            'tables_extracted': ['sum', 'mean'],
            'total_cells': ['sum', 'mean'],
            'avg_rows_per_table': 'mean',
            'avg_cols_per_table': 'mean',
            'execution_time': ['sum', 'mean'],
            'memory_used_mb': ['sum', 'mean']
        }).round(3)
        
        print(summary)

        print(f"\n{'='*80}")
        print("DETAILED RESULTS - Per File")
        print(f"{'='*80}\n")
        
        for filename in df_results['filename'].unique():
            file_data = df_results[df_results['filename'] == filename]
            print(f"\n📄 {filename}")
            print("-" * 80)
            
            display_df = file_data[['library', 'tables_extracted', 'total_cells', 
                                    'avg_rows_per_table', 'avg_cols_per_table', 
                                    'execution_time', 'memory_used_mb', 'error']].copy()
            print(display_df.to_string(index=False))

        csv_path = os.path.join(self.output_dir, f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df_results.to_csv(csv_path, index=False)
        print(f"\n✅ Results saved to: {csv_path}")

        json_path = os.path.join(self.output_dir, f"benchmark_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        summary_dict = {
            'timestamp': datetime.now().isoformat(),
            'total_files_processed': len(df_results['filename'].unique()),
            'libraries_tested': df_results['library'].unique().tolist(),
            'results': [asdict(r) for r in self.results]
        }
        with open(json_path, 'w') as f:
            json.dump(summary_dict, f, indent=2, default=str)
        print(f"✅ Summary saved to: {json_path}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "./input-docs"
    
    benchmark = PDFTableBenchmark()
    
    if os.path.isfile(path) and path.endswith('.pdf'):
        benchmark.benchmark_file(path)
    elif os.path.isdir(path):
        benchmark.benchmark_directory(path)
    else:
        print(f"Error: {path} is not a valid PDF file or directory")
        sys.exit(1)
    
    benchmark.generate_report()
