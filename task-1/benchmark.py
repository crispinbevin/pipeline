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
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results: List[ExtractionMetrics] = []

    # PyMuPDF extraction
    def extract_with_pymupdf(self, file_path: str) -> Tuple[int, List[pd.DataFrame], float, float]:
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

    # Camelot extraction
    def extract_with_camelot(self, file_path: str) -> Tuple[int, List[pd.DataFrame], float, float]:
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

    # pdfplumber extraction
    def extract_with_pdfplumber(self, file_path: str) -> Tuple[int, List[pd.DataFrame], float, float]:
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

    # Calculate aggregated metrics
    def calculate_metrics(self, library: str, filename: str, table_count: int, 
                         tables: List[pd.DataFrame], exec_time: float, memory_mb: float) -> ExtractionMetrics:
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

    # Save extracted tables to CSV
    def save_tables_to_csv(self, tables: List[pd.DataFrame], library: str, filename: str, page_num: int = None):
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

    # Benchmark a single file across all libraries
    def benchmark_file(self, file_path: str):
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
                
                # Save extracted tables to CSV
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

    # Benchmark directory
    def benchmark_directory(self, directory_path: str):
        pdf_files = [f for f in os.listdir(directory_path) if f.endswith('.pdf')]
        
        if not pdf_files:
            print(f"No PDF files found in {directory_path}")
            return

        print(f"Found {len(pdf_files)} PDF file(s)")
        
        for pdf_file in pdf_files:
            self.benchmark_file(os.path.join(directory_path, pdf_file))

    # Generate comparison report
    def generate_report(self):
        if not self.results:
            print("No results to report")
            return

        # Convert to DataFrame
        df_results = pd.DataFrame([asdict(r) for r in self.results])
        
        # Summary statistics grouped by library
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

        # Per-file comparison
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

        # Save to CSV
        csv_path = os.path.join(self.output_dir, f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df_results.to_csv(csv_path, index=False)
        print(f"\n✅ Results saved to: {csv_path}")

        # Save summary stats
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


# Usage
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