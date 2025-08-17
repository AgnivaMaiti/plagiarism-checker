import pandas as pd
import numpy as np
import xgboost as xgb
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# Import our utility functions
from utils.text_utils import get_tfidf_score, char_ngram_jaccard, normalized_levenshtein


class PlagiarismAnalyzer:
    """
    The detailed analysis engine. Handles model training and 1-to-1 comparison.
    """
    def __init__(self):
        # Initialize all model components
        self.tfidf_vectorizer = TfidfVectorizer()
        self.sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.scoring_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        self.scaler = StandardScaler()
        print("✅ Plagiarism Analyzer initialized.")

    def train(self, training_data_path='plagiarism_dataset.csv'):
        """Trains all the components of the detailed analysis engine."""
        print("  - Training the analysis engine... This may take a couple of minutes.")
        try:
            df = pd.read_csv(training_data_path)
            df.dropna(subset=['original', 'plagiarized'], inplace=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"Training data not found at {training_data_path}.")

        # 1. Train TF-IDF Vectorizer
        corpus = pd.concat([df['original'], df['plagiarized']]).unique()
        self.tfidf_vectorizer.fit(corpus)
        print("  - TF-IDF Vectorizer is ready.")

        # 2. Train the Scoring Model with a balanced, non-redundant feature set
        print("  - Generating a simplified feature set for the scoring model...")
        mono_df = df.sample(frac=0.5, random_state=42)
        features = self._featurize_dataframe(mono_df)
        positive_features = features.values

        print("  - Generating a large, balanced set of cross-lingual examples...")
        translator = pipeline("translation_en_to_de", model="Helsinki-NLP/opus-mt-en-de", device=0)
        cross_lingual_base_df = df.drop(mono_df.index)
        translated_texts = [result['translation_text'] for result in translator(cross_lingual_base_df['original'].tolist())]
        cross_lingual_df = pd.DataFrame({'original': cross_lingual_base_df['original'].tolist(), 'plagiarized': translated_texts})
        xl_features = self._featurize_dataframe(cross_lingual_df, is_cross_lingual=True)
        cross_lingual_positive_features = xl_features.values

        final_positive_features = np.vstack((positive_features, cross_lingual_positive_features))
        positive_labels = np.ones(len(final_positive_features))
        
        num_negative = len(final_positive_features)
        neg_pairs = pd.DataFrame({'original': df['original'].sample(n=num_negative, random_state=1, replace=True).values, 'plagiarized': df['original'].sample(n=num_negative, random_state=2, replace=True).values})
        neg_pairs = neg_pairs[neg_pairs['original'] != neg_pairs['plagiarized']]
        negative_features = self._featurize_dataframe(neg_pairs).values
        negative_labels = np.zeros(len(neg_pairs))

        X_train = np.vstack((final_positive_features, negative_features))
        y_train = np.concatenate((positive_labels, negative_labels))
        
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        self.scoring_model.fit(X_train_scaled, y_train)
        print("  - Final, Robust Scoring Model (XGBoost) is ready.")
        
    def _featurize_dataframe(self, df, is_cross_lingual=False):
        """Helper function to generate all features for a dataframe of text pairs."""
        features = pd.DataFrame()
        if is_cross_lingual:
            features['lexical_tfidf'] = 0.0
        else:
            features['lexical_tfidf'] = df.apply(lambda r: get_tfidf_score(self.tfidf_vectorizer, r['original'], r['plagiarized']), axis=1)

        multi_orig_emb = self.sbert_model.encode(df['original'].tolist())
        multi_plag_emb = self.sbert_model.encode(df['plagiarized'].tolist())
        features['semantic_multi'] = np.diag(util.cos_sim(multi_orig_emb, multi_plag_emb))
        
        features['lexical_char_ngram'] = df.apply(lambda r: char_ngram_jaccard(r['original'], r['plagiarized']), axis=1)
        features['lexical_levenshtein'] = df.apply(lambda r: normalized_levenshtein(r['original'], r['plagiarized']), axis=1)
        return features

    def check(self, text1, text2):
        """Checks two pieces of text for plagiarism using the final, hierarchical system."""
        # Step 1: Calculate all component scores
        lexical_tfidf = get_tfidf_score(self.tfidf_vectorizer, text1, text2)
        multi_embeddings = self.sbert_model.encode([text1, text2])
        semantic_multi = util.cos_sim(multi_embeddings[0], multi_embeddings[1]).item()
        lexical_char_ngram = char_ngram_jaccard(text1, text2)
        lexical_levenshtein = normalized_levenshtein(text1, text2)

        # Step 2: Implement the Hierarchical Gating Logic
        final_prob = 0.0
        detection_method = "Hybrid ML Model."

        if semantic_multi > 0.75 and lexical_tfidf < 0.25:
            final_prob = 0.90 + (semantic_multi - 0.75) * 0.4
            detection_method = "Expert Rule: High semantic similarity with low lexical overlap detected (likely cross-lingual)."
        else:
            features = np.array([[lexical_tfidf, semantic_multi, lexical_char_ngram, lexical_levenshtein]])
            features_scaled = self.scaler.transform(features)
            final_prob = self.scoring_model.predict_proba(features_scaled)[0][1]

        # Step 3: Format and return the final, explainable result
        return {
            'score_prob': final_prob,
            'unified_plagiarism_score': f"{final_prob * 100:.2f}%",
            'detection_method': detection_method,
            'component_scores': {
                'lexical_word_similarity (TF-IDF)': f"{lexical_tfidf:.4f}",
                'lexical_char_similarity (N-gram)': f"{lexical_char_ngram:.4f}",
                'lexical_edit_similarity (Levenshtein)': f"{lexical_levenshtein:.4f}",
                'semantic_similarity (Multilingual)': f"{semantic_multi:.4f}"
            }
        }