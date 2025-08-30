# engine/analysis_engine.py (FINAL, CORRECTED VERSION)

import pandas as pd
import numpy as np
import xgboost as xgb
import logging
import joblib # <-- Import joblib
import os
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from .text_utils import get_tfidf_score, char_ngram_jaccard, normalized_levenshtein

class PlagiarismAnalyzer:
    def __init__(self, model_path='models'):
        self.model_path = model_path
        self.tfidf_vectorizer = TfidfVectorizer()
        self.sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.scoring_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        self.scaler = StandardScaler()
        os.makedirs(self.model_path, exist_ok=True) # Ensure model directory exists
        logging.info("Plagiarism Analyzer initialized with all components.")

    def save(self):
        """Saves the trained components to disk."""
        logging.info(f"Saving trained models to '{self.model_path}'...")
        joblib.dump(self.tfidf_vectorizer, os.path.join(self.model_path, 'tfidf_vectorizer.joblib'))
        joblib.dump(self.scaler, os.path.join(self.model_path, 'scaler.joblib'))
        joblib.dump(self.scoring_model, os.path.join(self.model_path, 'scoring_model.joblib'))
        logging.info("Models saved successfully.")

    def load(self):
        """Loads the trained components from disk."""
        logging.info(f"Loading trained models from '{self.model_path}'...")
        try:
            self.tfidf_vectorizer = joblib.load(os.path.join(self.model_path, 'tfidf_vectorizer.joblib'))
            self.scaler = joblib.load(os.path.join(self.model_path, 'scaler.joblib'))
            self.scoring_model = joblib.load(os.path.join(self.model_path, 'scoring_model.joblib'))
            logging.info("Models loaded successfully.")
            return True
        except FileNotFoundError:
            logging.error("Could not load models. Files not found. Please run the training first.")
            return False

    def train(self, training_data_path='plagiarism_dataset.csv'):
        """Trains all the components of the detailed analysis engine."""
        logging.info("Starting analysis engine training...")
        try:
            df = pd.read_csv(training_data_path)
            df.dropna(subset=['original', 'plagiarized'], inplace=True)
            logging.info(f"Training data loaded successfully. Shape: {df.shape}")
        except FileNotFoundError:
            logging.error(f"Training data not found at {training_data_path}.")
            raise

        logging.info("Training TF-IDF Vectorizer...")
        corpus = pd.concat([df['original'], df['plagiarized']]).unique()
        self.tfidf_vectorizer.fit(corpus)
        logging.info("TF-IDF Vectorizer is ready.")

        logging.info("Generating features for the scoring model...")
        mono_df = df.sample(frac=0.5, random_state=42)
        features = self._featurize_dataframe(mono_df)
        positive_features = features.values
        logging.info(f"Generated {len(positive_features)} monolingual positive features.")

        logging.warning("Cross-lingual training step is DISABLED to save memory.")
        final_positive_features = positive_features
        positive_labels = np.ones(len(final_positive_features))
        
        logging.info("Generating negative (non-plagiarized) examples...")
        num_negative = len(final_positive_features)
        neg_pairs = pd.DataFrame({'original': df['original'].sample(n=num_negative, random_state=1, replace=True).values, 'plagiarized': df['original'].sample(n=num_negative, random_state=2, replace=True).values})
        neg_pairs = neg_pairs[neg_pairs['original'] != neg_pairs['plagiarized']]
        negative_features = self._featurize_dataframe(neg_pairs).values
        negative_labels = np.zeros(len(neg_pairs))

        X_train = np.vstack((final_positive_features, negative_features))
        y_train = np.concatenate((positive_labels, negative_labels))
        
        logging.info("Fitting the StandardScaler...")
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        
        logging.info("Training the final XGBoost scoring model...")
        self.scoring_model.fit(X_train_scaled, y_train)
        logging.info("Scoring Model (XGBoost) is ready.")
        
    def _featurize_dataframe(self, df):
        features = pd.DataFrame()
        features['lexical_tfidf'] = df.apply(lambda r: get_tfidf_score(self.tfidf_vectorizer, r['original'], r['plagiarized']), axis=1)
        multi_orig_emb = self.sbert_model.encode(df['original'].tolist())
        multi_plag_emb = self.sbert_model.encode(df['plagiarized'].tolist())
        features['semantic_multi'] = np.diag(util.cos_sim(multi_orig_emb, multi_plag_emb))
        features['lexical_char_ngram'] = df.apply(lambda r: char_ngram_jaccard(r['original'], r['plagiarized']), axis=1)
        features['lexical_levenshtein'] = df.apply(lambda r: normalized_levenshtein(r['original'], r['plagiarized']), axis=1)
        return features

    def check(self, text1, text2):
        lexical_tfidf = get_tfidf_score(self.tfidf_vectorizer, text1, text2)
        multi_embeddings = self.sbert_model.encode([text1, text2])
        semantic_multi = util.cos_sim(multi_embeddings[0], multi_embeddings[1]).item()
        lexical_char_ngram = char_ngram_jaccard(text1, text2)
        lexical_levenshtein = normalized_levenshtein(text1, text2)
        
        final_prob = 0.0
        detection_method = "Hybrid ML Model."

        if semantic_multi > 0.75 and lexical_tfidf < 0.25:
            final_prob = 0.90 + (semantic_multi - 0.75) * 0.4
            detection_method = "Expert Rule: High semantic similarity with low lexical overlap detected."
        else:
            features = np.array([[lexical_tfidf, semantic_multi, lexical_char_ngram, lexical_levenshtein]])
            features_scaled = self.scaler.transform(features)
            final_prob = self.scoring_model.predict_proba(features_scaled)[0][1]
        
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