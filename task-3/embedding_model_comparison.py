import json
import os
import sys
import argparse
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine, euclidean
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingComparator:
    """Compare performance of different embedding models"""
    
    def __init__(self, jsonl_path, num_samples=500, output_dir="./comparison_results"):
        self.jsonl_path = jsonl_path
        self.num_samples = num_samples
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.validation_texts = self._load_validation_data()
        self.results = {}
        
    def _load_validation_data(self):
        """Load validation samples from JSONL"""
        texts = []
        with open(self.jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                text = data.get('text') or data.get('content') or str(data)
                if text and len(text.strip()) > 0:
                    texts.append(text)
                if len(texts) >= self.num_samples:
                    break
        
        logger.info(f"Loaded {len(texts)} validation samples")
        return texts
    
    def generate_embeddings(self, model_name_or_path, device='cuda'):
        """Generate embeddings using a model"""
        logger.info(f"Loading model: {model_name_or_path}")
        try:
            model = SentenceTransformer(model_name_or_path, device=device)
            logger.info(f"Generating embeddings with {model_name_or_path}...")
            embeddings = model.encode(self.validation_texts, show_progress_bar=True, convert_to_numpy=True)
            return embeddings, model.get_sentence_embedding_dimension()
        except Exception as e:
            logger.error(f"Failed to load {model_name_or_path}: {e}")
            return None, None
    
    def compute_similarity_matrix(self, embeddings):
        """Compute cosine similarity matrix"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / norms
        similarity_matrix = np.dot(normalized, normalized.T)
        return similarity_matrix
    
    def evaluate_retrieval_quality(self, embeddings):
        """Evaluate retrieval quality: how well consecutive chunks are similar"""
        similarity_matrix = self.compute_similarity_matrix(embeddings)
        
        # Adjacent chunks should be similar (they're consecutive in document)
        adjacent_similarities = []
        for i in range(len(embeddings) - 1):
            adjacent_similarities.append(similarity_matrix[i, i+1])
        
        mean_adjacent = np.mean(adjacent_similarities)
        median_adjacent = np.median(adjacent_similarities)
        std_adjacent = np.std(adjacent_similarities)
        
        return {
            'mean_adjacent_similarity': float(mean_adjacent),
            'median_adjacent_similarity': float(median_adjacent),
            'std_adjacent_similarity': float(std_adjacent),
        }
    
    def evaluate_clustering_quality(self, embeddings, n_clusters=10):
        """Evaluate clustering quality using multiple metrics"""
        logger.info(f"Computing clustering metrics with {n_clusters} clusters...")
        
        kmeans = KMeans(n_clusters=min(n_clusters, len(embeddings)), random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        # Silhouette Score (higher is better, range: -1 to 1)
        silhouette = silhouette_score(embeddings, labels)
        
        # Davies-Bouldin Index (lower is better, range: 0 to inf)
        davies_bouldin = davies_bouldin_score(embeddings, labels)
        
        # Calinski-Harabasz Index (higher is better, range: 0 to inf)
        calinski_harabasz = calinski_harabasz_score(embeddings, labels)
        
        return {
            'silhouette_score': float(silhouette),
            'davies_bouldin_index': float(davies_bouldin),
            'calinski_harabasz_index': float(calinski_harabasz),
        }
    
    def evaluate_embedding_properties(self, embeddings):
        """Evaluate general embedding properties"""
        # Check for NaN or Inf
        has_nan = np.isnan(embeddings).any()
        has_inf = np.isinf(embeddings).any()
        
        # Embedding statistics
        magnitude_mean = float(np.mean(np.linalg.norm(embeddings, axis=1)))
        magnitude_std = float(np.std(np.linalg.norm(embeddings, axis=1)))
        
        # Variance per dimension
        var_per_dim = float(np.mean(np.var(embeddings, axis=0)))
        
        # Orthogonality (correlation between random pairs)
        random_pairs_corr = []
        for _ in range(min(100, len(embeddings) // 2)):
            i, j = np.random.choice(len(embeddings), 2, replace=False)
            corr = np.dot(embeddings[i], embeddings[j]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-8)
            random_pairs_corr.append(abs(corr))
        
        mean_random_corr = float(np.mean(random_pairs_corr))
        
        return {
            'has_nan': bool(has_nan),
            'has_inf': bool(has_inf),
            'embedding_magnitude_mean': magnitude_mean,
            'embedding_magnitude_std': magnitude_std,
            'variance_per_dimension': var_per_dim,
            'mean_random_correlation': mean_random_corr,
        }
    
    def compare_models(self, model_list, n_clusters=10, device='cuda'):
        """Compare multiple models"""
        logger.info(f"Comparing {len(model_list)} models on {len(self.validation_texts)} samples")
        
        for i, model_name in enumerate(model_list, 1):
            logger.info(f"\n[{i}/{len(model_list)}] Evaluating: {model_name}")
            
            embeddings, embedding_dim = self.generate_embeddings(model_name, device)
            
            if embeddings is None:
                logger.warning(f"Skipped {model_name}")
                continue
            
            # Compute all metrics
            retrieval_metrics = self.evaluate_retrieval_quality(embeddings)
            clustering_metrics = self.evaluate_clustering_quality(embeddings, n_clusters)
            embedding_props = self.evaluate_embedding_properties(embeddings)
            
            self.results[model_name] = {
                'embedding_dimension': embedding_dim,
                'num_samples': len(self.validation_texts),
                'retrieval': retrieval_metrics,
                'clustering': clustering_metrics,
                'properties': embedding_props,
            }
            
            logger.info(f"✓ Completed {model_name}")
        
        return self.results
    
    def generate_report(self):
        """Generate text-based comparison report"""
        if not self.results:
            logger.error("No results to report")
            return
        
        report_path = self.output_dir / "evaluation_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("EMBEDDING MODEL PERFORMANCE COMPARISON REPORT\n")
            f.write("=" * 100 + "\n\n")
            
            f.write(f"Evaluation Date: {datetime.now().isoformat()}\n")
            f.write(f"Validation Samples: {len(self.validation_texts)}\n")
            f.write(f"Models Compared: {len(self.results)}\n")
            f.write(f"Input File: {self.jsonl_path}\n\n")
            
            # Summary table
            f.write("-" * 100 + "\n")
            f.write("SUMMARY TABLE\n")
            f.write("-" * 100 + "\n\n")
            
            f.write(f"{'Model':<40} {'Dim':<8} {'Silhouette':<15} {'Davies-Bouldin':<15} {'Mean Sim':<12}\n")
            f.write("-" * 100 + "\n")
            
            for model_name, metrics in sorted(self.results.items()):
                dim = metrics['embedding_dimension']
                silhouette = metrics['clustering']['silhouette_score']
                db = metrics['clustering']['davies_bouldin_index']
                mean_sim = metrics['retrieval']['mean_adjacent_similarity']
                f.write(f"{model_name:<40} {dim:<8} {silhouette:<15.4f} {db:<15.4f} {mean_sim:<12.4f}\n")
            
            f.write("\n")
            
            # Detailed metrics for each model
            f.write("=" * 100 + "\n")
            f.write("DETAILED EVALUATION METRICS BY MODEL\n")
            f.write("=" * 100 + "\n\n")
            
            for model_name, metrics in sorted(self.results.items()):
                f.write(f"\nMODEL: {model_name}\n")
                f.write("-" * 100 + "\n")
                
                f.write(f"Embedding Dimension: {metrics['embedding_dimension']}\n")
                f.write(f"Samples Evaluated: {metrics['num_samples']}\n\n")
                
                # Retrieval metrics
                f.write("RETRIEVAL QUALITY (Semantic Similarity of Adjacent Chunks):\n")
                f.write(f"  Mean Adjacent Similarity:        {metrics['retrieval']['mean_adjacent_similarity']:.4f}\n")
                f.write(f"  Median Adjacent Similarity:      {metrics['retrieval']['median_adjacent_similarity']:.4f}\n")
                f.write(f"  Std Dev Adjacent Similarity:     {metrics['retrieval']['std_adjacent_similarity']:.4f}\n")
                f.write(f"  → Higher is better (range: -1 to 1)\n\n")
                
                # Clustering metrics
                f.write("CLUSTERING QUALITY:\n")
                f.write(f"  Silhouette Score:                {metrics['clustering']['silhouette_score']:.4f}\n")
                f.write(f"    → Range: -1 to 1 (higher is better, >0.5 is good)\n")
                f.write(f"  Davies-Bouldin Index:            {metrics['clustering']['davies_bouldin_index']:.4f}\n")
                f.write(f"    → Range: 0 to ∞ (lower is better, <1 is excellent)\n")
                f.write(f"  Calinski-Harabasz Index:         {metrics['clustering']['calinski_harabasz_index']:.4f}\n")
                f.write(f"    → Range: 0 to ∞ (higher is better)\n\n")
                
                # Embedding properties
                f.write("EMBEDDING PROPERTIES:\n")
                f.write(f"  Has NaN values:                  {metrics['properties']['has_nan']}\n")
                f.write(f"  Has Inf values:                  {metrics['properties']['has_inf']}\n")
                f.write(f"  Magnitude Mean:                  {metrics['properties']['embedding_magnitude_mean']:.4f}\n")
                f.write(f"  Magnitude Std Dev:               {metrics['properties']['embedding_magnitude_std']:.4f}\n")
                f.write(f"  Variance per Dimension:          {metrics['properties']['variance_per_dimension']:.6f}\n")
                f.write(f"  Mean Random Correlation:         {metrics['properties']['mean_random_correlation']:.4f}\n")
                f.write(f"    → Lower is better (indicates diversity)\n\n")
            
            # Rankings
            f.write("=" * 100 + "\n")
            f.write("RANKINGS BY METRIC\n")
            f.write("=" * 100 + "\n\n")
            
            # Rank by silhouette
            sorted_by_silhouette = sorted(
                self.results.items(),
                key=lambda x: x[1]['clustering']['silhouette_score'],
                reverse=True
            )
            f.write("TOP 5 BY SILHOUETTE SCORE (Clustering Quality):\n")
            for i, (name, metrics) in enumerate(sorted_by_silhouette[:5], 1):
                f.write(f"  {i}. {name:<50} {metrics['clustering']['silhouette_score']:.4f}\n")
            f.write("\n")
            
            # Rank by Davies-Bouldin
            sorted_by_db = sorted(
                self.results.items(),
                key=lambda x: x[1]['clustering']['davies_bouldin_index']
            )
            f.write("TOP 5 BY DAVIES-BOULDIN INDEX (Lower is Better):\n")
            for i, (name, metrics) in enumerate(sorted_by_db[:5], 1):
                f.write(f"  {i}. {name:<50} {metrics['clustering']['davies_bouldin_index']:.4f}\n")
            f.write("\n")
            
            # Rank by adjacent similarity
            sorted_by_sim = sorted(
                self.results.items(),
                key=lambda x: x[1]['retrieval']['mean_adjacent_similarity'],
                reverse=True
            )
            f.write("TOP 5 BY RETRIEVAL QUALITY (Adjacent Similarity):\n")
            for i, (name, metrics) in enumerate(sorted_by_sim[:5], 1):
                f.write(f"  {i}. {name:<50} {metrics['retrieval']['mean_adjacent_similarity']:.4f}\n")
            f.write("\n")
            
            # Recommendations
            f.write("=" * 100 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("=" * 100 + "\n\n")
            
            best_silhouette = sorted_by_silhouette[0]
            best_db = sorted_by_db[0]
            best_retrieval = sorted_by_sim[0]
            
            f.write(f"Best for Clustering (Silhouette):    {best_silhouette[0]}\n")
            f.write(f"Best for Cluster Separation (DB):    {best_db[0]}\n")
            f.write(f"Best for Retrieval Quality:          {best_retrieval[0]}\n\n")
            
            f.write("Overall Recommendation:\n")
            if best_silhouette[1]['clustering']['silhouette_score'] > 0.5:
                f.write(f"  Use {best_silhouette[0]} for your RAG pipeline.\n")
            else:
                f.write(f"  Consider {best_silhouette[0]}, but validate on your use case.\n")
            f.write(f"  (Silhouette > 0.5 indicates good cluster structure)\n\n")
            
            f.write("=" * 100 + "\n")
        
        logger.info(f"Report saved to {report_path}")
        return report_path
    
    def save_results_json(self):
        """Save detailed results as JSON"""
        json_path = self.output_dir / "detailed_results.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Detailed results saved to {json_path}")
        return json_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare embedding model performance")
    parser.add_argument("jsonl_path", help="Path to validation JSONL file")
    parser.add_argument("--models", nargs="+", required=True, 
                       help="List of model names to compare (e.g., all-MiniLM-L6-v2 all-mpnet-base-v2)")
    parser.add_argument("--num-samples", type=int, default=500,
                       help="Number of validation samples to use (default: 500)")
    parser.add_argument("--n-clusters", type=int, default=10,
                       help="Number of clusters for clustering evaluation (default: 10)")
    parser.add_argument("--output-dir", default="./comparison_results",
                       help="Output directory for results (default: ./comparison_results)")
    parser.add_argument("--device", default="cuda",
                       help="Device to use: cuda or cpu (default: cuda)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.jsonl_path):
        print(f"Error: {args.jsonl_path} not found!")
        sys.exit(1)
    
    comparator = EmbeddingComparator(
        jsonl_path=args.jsonl_path,
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )
    
    results = comparator.compare_models(
        model_list=args.models,
        n_clusters=args.n_clusters,
        device=args.device
    )
    
    comparator.generate_report()
    comparator.save_results_json()
    
    print(f"\n✓ Comparison complete! Results saved to {args.output_dir}")