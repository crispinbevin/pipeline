import json
import os
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from sentence_transformers.evaluation import InformationRetrievalEvaluator
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChunkDataset(Dataset):
    """Load chunked documents from JSONL for training"""
    def __init__(self, jsonl_path):
        self.examples = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                # Expects: {"text": "...", "metadata": {...}} or just {"text": "..."}
                text = data.get('text') or data.get('content') or str(data)
                if text and len(text.strip()) > 0:
                    self.examples.append(text)
        logger.info(f"Loaded {len(self.examples)} chunks from {jsonl_path}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def create_training_pairs(texts, window_size=3):
    """Create sentence pairs for contrastive learning from consecutive chunks"""
    pairs = []
    for i in range(len(texts) - 1):
        # Positive pair: consecutive chunks (semantically related)
        pairs.append(InputExample(texts=[texts[i], texts[i+1]], label=0.8))
        
        # Add same-text pair for strong signal
        if len(texts[i]) > 50:
            pairs.append(InputExample(texts=[texts[i], texts[i]], label=1.0))
    
    logger.info(f"Created {len(pairs)} training pairs")
    return pairs

def train_embedding_model(
    jsonl_path,
    output_dir="./fine_tuned_embeddings",
    model_name="all-MiniLM-L6-v2",  # Lightweight, ~22M params
    epochs=3,
    batch_size=32,
    warmup_steps=100,
    device=None
):
    """
    Fine-tune embedding model on your chunked data
    
    Args:
        jsonl_path: Path to your JSONL file
        output_dir: Where to save model weights
        model_name: Pretrained model from sentence-transformers
        epochs: Training epochs
        batch_size: Batch size (adjust based on GPU memory)
        warmup_steps: Learning rate warmup
        device: 'cuda' or 'cpu' (auto-detect if None)
    """
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Create output directories
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logs_dir = Path(output_dir) / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Load data
    dataset = ChunkDataset(jsonl_path)
    texts = dataset.examples
    
    if len(texts) == 0:
        raise ValueError("No valid chunks found in JSONL file")
    
    # Create training pairs
    train_examples = create_training_pairs(texts)
    
    # Initialize model
    logger.info(f"Loading pretrained model: {model_name}")
    model = SentenceTransformer(model_name, device=device)
    
    # Setup training
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    
    # Use MultipleNegativesRankingLoss for efficient contrastive learning
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # Training parameters
    num_epochs = epochs
    num_steps = len(train_dataloader) * num_epochs
    
    logger.info(f"Training for {num_epochs} epochs, {num_steps} total steps")
    
    # Train
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        output_path=output_dir,
        checkpoint_save_steps=len(train_dataloader),
        checkpoint_save_total_limit=2,
        show_progress_bar=True,
    )
    
    # Save final model
    final_model_path = Path(output_dir) / "final_model"
    model.save(str(final_model_path))
    logger.info(f"Model saved to {final_model_path}")
    
    # Generate example embeddings
    logger.info("Generating example embeddings...")
    sample_texts = texts[:min(100, len(texts))]
    embeddings = model.encode(sample_texts, convert_to_tensor=False, show_progress_bar=True)
    
    # Save embeddings and metrics
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "total_chunks": len(texts),
        "training_pairs": len(train_examples),
        "epochs": num_epochs,
        "batch_size": batch_size,
        "embedding_dimension": embeddings.shape[1],
        "device": device
    }
    
    with open(logs_dir / "training_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save sample embeddings
    np.save(logs_dir / "sample_embeddings.npy", embeddings)
    with open(logs_dir / "sample_texts.jsonl", 'w') as f:
        for i, text in enumerate(sample_texts):
            f.write(json.dumps({
                "index": i,
                "text": text[:500],  # First 500 chars
                "embedding_shape": list(embeddings[i].shape)
            }) + '\n')
    
    logger.info(f"Metrics saved to {logs_dir / 'training_metrics.json'}")
    logger.info(f"Sample embeddings saved to {logs_dir / 'sample_embeddings.npy'}")
    
    return model, metrics

def load_finetuned_model(model_path):
    """Load the fine-tuned model for inference"""
    return SentenceTransformer(model_path)

def generate_embeddings(model, texts, batch_size=32):
    """Generate embeddings for texts"""
    return model.encode(texts, batch_size=batch_size, convert_to_tensor=False)

if __name__ == "__main__":
    # Configuration
    JSONL_PATH = "../pre-processed-docs.jsonl"  # Your JSONL in parent directory
    OUTPUT_DIR = "./fine_tuned_embeddings"
    MODEL_NAME = "all-MiniLM-L6-v2"  # 22M params, great for GPU with 16GB RAM
    EPOCHS = 3
    BATCH_SIZE = 32  # Adjust if OOM: try 16 or 8
    
    # Verify JSONL exists
    if not os.path.exists(JSONL_PATH):
        print(f"Error: {JSONL_PATH} not found!")
        print("Please update JSONL_PATH to point to your chunked data file")
        exit(1)
    
    # Train
    model, metrics = train_embedding_model(
        jsonl_path=JSONL_PATH,
        output_dir=OUTPUT_DIR,
        model_name=MODEL_NAME,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        warmup_steps=100
    )
    
    print("\n✓ Training complete!")
    print(f"Model saved to: {OUTPUT_DIR}/final_model")
    print(f"Metrics: {metrics}")
    
    # Test inference
    print("\nTesting inference on first 5 samples...")
    test_texts = ["sample text 1", "sample text 2"]
    embeddings = generate_embeddings(model, test_texts)
    print(f"Generated embeddings shape: {embeddings.shape}")