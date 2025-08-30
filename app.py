# app.py (FINAL, CORRECTED VERSION)

import os
import sys
import logging
import pandas as pd
from flask import Flask, render_template, request, flash, redirect, url_for

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from engine.analysis_engine import PlagiarismAnalyzer
from engine.indexer import Indexer
from engine.search_engine import SearchEngine

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# --- CONFIGURATION ---
CORPUS_PATH = "corpus"
INDEX_PATH = "corpus.faiss"
METADATA_PATH = "metadata.json"
TRAINING_DATA_PATH = 'plagiarism_dataset.csv'
MODEL_PATH = 'models' # Path to save/load trained models

search_engine = None

def create_corpus_from_dataset(dataset_path='plagiarism_dataset.csv', corpus_path="corpus"):
    logging.info(f"Creating document corpus from '{dataset_path}'...")
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        logging.error(f"Dataset not found at {dataset_path}.")
        return False
    if 'original' not in df.columns:
        logging.error("'original' column not found in dataset.")
        return False
    os.makedirs(corpus_path, exist_ok=True)
    source_texts = df['original'].unique().tolist()
    for i, text in enumerate(source_texts):
        with open(os.path.join(corpus_path, f"doc_{i+1}.txt"), "w", encoding='utf-8') as f:
            f.write(str(text))
    logging.info(f"Successfully created {len(source_texts)} documents in the corpus.")
    return True

def initialize_engine():
    """Performs the one-time setup: training, saving models, corpus creation, and indexing."""
    global search_engine
    logging.info("--- Starting One-Time Engine Initialization ---")
    try:
        analyzer = PlagiarismAnalyzer(model_path=MODEL_PATH)
        analyzer.train(training_data_path=TRAINING_DATA_PATH)
        analyzer.save() # <-- SAVE THE TRAINED MODELS
        logging.info("Analyzer training and model saving completed successfully.")

        if not create_corpus_from_dataset(dataset_path=TRAINING_DATA_PATH, corpus_path=CORPUS_PATH):
            raise RuntimeError("Failed to create the corpus from the dataset.")

        indexer = Indexer(sbert_model=analyzer.sbert_model)
        indexer.build_index(corpus_path=CORPUS_PATH)
        indexer.save(index_path=INDEX_PATH, meta_path=METADATA_PATH)
        logging.info("Search index built and saved successfully.")

        search_engine = SearchEngine(analysis_engine=analyzer)
        search_engine.load_index(
            index_path=INDEX_PATH, meta_path=METADATA_PATH, corpus_path=CORPUS_PATH
        )
        logging.info("--- ENGINE IS READY ---")
    except Exception:
        logging.error("!!!!!! A FATAL ERROR OCCURRED DURING ENGINE INITIALIZATION !!!!!!", exc_info=True)
        search_engine = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check', methods=['POST'])
def check_plagiarism():
    if search_engine is None:
        flash("CRITICAL ERROR: The plagiarism engine is not initialized.", "danger")
        return redirect(url_for('index'))
    query_text = request.form.get('query_text')
    if not query_text or not query_text.strip():
        flash("Please enter some text to check.", "warning")
        return redirect(url_for('index'))
    sources = search_engine.find_plagiarism_sources(query_text)
    return render_template('results.html', query=query_text, sources=sources)

@app.route('/compare', methods=['GET'])
def compare_page():
    return render_template('compare.html')

@app.route('/compare', methods=['POST'])
def perform_compare():
    if search_engine is None:
        flash("CRITICAL ERROR: The plagiarism engine is not initialized.", "danger")
        return redirect(url_for('compare_page'))
    text1 = request.form.get('text1')
    text2 = request.form.get('text2')
    if not text1 or not text2:
        flash("Please provide text in both boxes for comparison.", "warning")
        return redirect(url_for('compare_page'))
    result = search_engine.analysis_engine.check(text1, text2)
    return render_template('compare_results.html', result=result, text1=text1, text2=text2)

if __name__ == '__main__':
    logging.info("Application starting up...")
    with app.app_context():
        # Check if either the index OR the models are missing to trigger setup
        if not os.path.exists(INDEX_PATH) or not os.path.exists(os.path.join(MODEL_PATH, 'scaler.joblib')):
            logging.warning("Search index or trained models not found. Running one-time setup...")
            initialize_engine()
        else:
            logging.info("Found existing index and models. Attempting to load engine...")
            try:
                analyzer = PlagiarismAnalyzer(model_path=MODEL_PATH)
                if not analyzer.load(): # <-- LOAD THE TRAINED MODELS
                    raise RuntimeError("Failed to load trained models.")
                
                search_engine = SearchEngine(analysis_engine=analyzer)
                if not search_engine.load_index(
                    index_path=INDEX_PATH, meta_path=METADATA_PATH, corpus_path=CORPUS_PATH
                ):
                     raise RuntimeError("Failed to load the search index from files.")
                logging.info("--- ENGINE IS READY ---")
            except Exception:
                logging.error("Could not load the existing engine due to an error.", exc_info=True)
                search_engine = None
    if search_engine is not None:
        logging.info("Starting Flask server...")
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    else:
        logging.critical("Flask server NOT started because the engine failed to initialize.")