import os
import json
import faiss
from sentence_transformers import SentenceTransformer
from .analysis_engine import PlagiarismAnalyzer # Import from within the same package

class SearchEngine:
    """Performs the two-stage search for plagiarism sources."""
    def __init__(self, analysis_engine: PlagiarismAnalyzer):
        self.analysis_engine = analysis_engine
        self.sbert_model = self.analysis_engine.sbert_model
        self.index = None
        self.metadata = {}
        self.corpus_path = ""

    def load_index(self, index_path="corpus.faiss", meta_path="metadata.json", corpus_path="corpus"):
        """Loads a pre-built FAISS index from disk."""
        print("\nLoading search index and metadata...")
        if not os.path.exists(index_path):
            print(f"❌ Index not found at {index_path}. Please build the index first.")
            return False
        self.index = faiss.read_index(index_path)
        with open(meta_path, "r") as f:
            self.metadata = {int(k): v for k, v in json.load(f).items()}
        self.corpus_path = corpus_path
        print("✅ Index loaded successfully.")
        return True

    def find_plagiarism_sources(self, query_text: str, top_k=5, score_threshold=0.5):
        """Performs the full 1-to-many plagiarism search."""
        print(f"\nSearching for plagiarism sources for the query text...")
        if self.index is None:
            print("❌ Index is not loaded. Cannot perform search.")
            return []

        # --- Stage 1: Candidate Retrieval ---
        print(f"  - Stage 1: Retrieving top {top_k} candidates from the vector database...")
        query_vector = self.sbert_model.encode([query_text])
        _, indices = self.index.search(query_vector, top_k)
        
        candidate_indices = indices[0]
        print(f"  - Retrieved candidates: {[self.metadata[i] for i in candidate_indices]}")

        # --- Stage 2: Detailed Analysis ---
        print("  - Stage 2: Performing detailed analysis on candidates...")
        results = []
        for idx in candidate_indices:
            filename = self.metadata[idx]
            try:
                with open(os.path.join(self.corpus_path, filename), "r", encoding='utf-8') as f:
                    candidate_text = f.read()
            except FileNotFoundError:
                print(f"  - Warning: Could not find candidate file {filename}. Skipping.")
                continue
            
            result = self.analysis_engine.check(query_text, candidate_text)
            
            if result['score_prob'] >= score_threshold:
                results.append({
                    "source_document": filename,
                    "details": result
                })
        
        # Sort results by score, descending
        results.sort(key=lambda x: x['details']['score_prob'], reverse=True)
        
        # Clean up the output
        for res in results:
            res['plagiarism_score'] = res['details']['unified_plagiarism_score']
            del res['details']['score_prob']
        
        print("✅ Search complete.")
        return results