# Embedding Model Fine-Tuning - Quick Start

## Install Dependencies

```bash
pip install sentence-transformers torch numpy
```

For CPU-only:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers numpy
```

## Run Fine-Tuning

### Basic Usage
```bash
python embedding_finetuning.py <jsonl_file>
```

### Example
```bash
python embedding_finetuning.py ../pre-processed-docs.jsonl
```

## Command Arguments

| Argument | Required | Default | Example |
|----------|----------|---------|---------|
| `<jsonl_file>` | ✓ | - | `../pre-processed-docs.jsonl` |
| `--output-dir` | ✗ | `./fine_tuned_embeddings` | `--output-dir ./embeddings` |
| `--model` | ✗ | `all-MiniLM-L6-v2` | `--model all-mpnet-base-v2` |
| `--epochs` | ✗ | 3 | `--epochs 5` |
| `--batch-size` | ✗ | 32 | `--batch-size 16` |
| `--warmup-steps` | ✗ | 100 | `--warmup-steps 200` |

## Common Examples

**Default (recommended):**
```bash
python embedding_finetuning.py ../pre-processed-docs.jsonl
```

**Custom output directory:**
```bash
python embedding_finetuning.py ../pre-processed-docs.jsonl \
  --output-dir ./my_embeddings \
  --epochs 5
```

**Better quality (slower):**
```bash
python embedding_finetuning.py ../pre-processed-docs.jsonl \
  --model all-mpnet-base-v2 \
  --epochs 5
```

**Faster training:**
```bash
python embedding_finetuning.py ../pre-processed-docs.jsonl \
  --epochs 2 \
  --batch-size 16
```

**Out of memory? Reduce batch size:**
```bash
python embedding_finetuning.py ../pre-processed-docs.jsonl \
  --batch-size 8
```

## Output

Model and results saved to `./fine_tuned_embeddings/`:
- `final_model/` - Trained model weights
- `logs/training_metrics.json` - Training statistics
- `logs/sample_embeddings.npy` - Example embeddings
- `logs/sample_texts.jsonl` - Corresponding texts

## Model Options

- `all-MiniLM-L6-v2` - Fast, 22M params, 384 dims (default)
- `all-mpnet-base-v2` - Best quality, 109M params, 768 dims
- `all-distilroberta-v1` - Balanced, 82M params, 768 dims

## Input Format

JSONL file with one JSON object per line:

```jsonl
{"text": "Document chunk content here..."}
{"text": "Another chunk..."}
{"text": "More content..."}
```

Or with metadata:
```jsonl
{"text": "Content...", "metadata": {"source": "doc1.pdf"}}
```


# Embedding Model Comparison - Quick Start

## Install Dependencies

```bash
pip install sentence-transformers scikit-learn scipy numpy
```

## Run Comparison

### Basic Usage
```bash
python embedding_model_comparison.py <jsonl_file> --models <model1> <model2> ...
```

### Example
```bash
python embedding_model_comparison.py ../pre-processed-docs.jsonl \
  --models all-MiniLM-L6-v2 all-mpnet-base-v2
```

## Command Arguments

| Argument | Required | Default | Example |
|----------|----------|---------|---------|
| `<jsonl_file>` | ✓ | - | `../pre-processed-docs.jsonl` |
| `--models` | ✓ | - | `all-MiniLM-L6-v2 all-mpnet-base-v2` |
| `--num-samples` | ✗ | 500 | `--num-samples 1000` |
| `--n-clusters` | ✗ | 10 | `--n-clusters 15` |
| `--output-dir` | ✗ | `./comparison_results` | `--output-dir ./results` |
| `--device` | ✗ | `cuda` | `--device cpu` |

## Common Examples

**Quick test (100 samples, 2 models):**
```bash
python embedding_model_comparison.py ../pre-processed-docs.jsonl \
  --models all-MiniLM-L6-v2 all-mpnet-base-v2 \
  --num-samples 100
```

**Full comparison (500 samples, 3 models):**
```bash
python embedding_model_comparison.py ../pre-processed-docs.jsonl \
  --models all-MiniLM-L6-v2 all-mpnet-base-v2 all-distilroberta-v1 \
  --num-samples 500
```

**CPU only:**
```bash
python embedding_model_comparison.py ../pre-processed-docs.jsonl \
  --models all-MiniLM-L6-v2 \
  --device cpu
```

## Results

Results are saved to `./comparison_results/`:
- `evaluation_report.txt` - Main report with metrics and rankings
- `detailed_results.json` - Detailed metrics in JSON format

View report:
```bash
cat comparison_results/evaluation_report.txt
```

## Model Options

- `all-MiniLM-L6-v2` - Fast, 384 dims
- `all-mpnet-base-v2` - Best quality, 768 dims
- `all-distilroberta-v1` - Balanced, 768 dims
- `./path/to/your/model` - Custom/fine-tuned models