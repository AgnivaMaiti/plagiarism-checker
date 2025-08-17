# Plagiarism Checker

A scalable, modular plagiarism detection system using semantic embeddings, machine learning, and a vector database (FAISS).  
Supports both monolingual and cross-lingual plagiarism detection.

---

## Features

- **Detailed 1-to-1 plagiarism analysis** using lexical, semantic, and ML-based features
- **Scalable 1-to-many search** using FAISS vector database
- **Cross-lingual detection** (e.g., English ↔ German)
- **Easy extensibility** for new datasets or document corpora

---

## Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies

---

## Installation

1. **Clone the repository:**

   ```sh
   git clone https://github.com/AgnivaMaiti/plagiarism-checker.git
   cd plagiarism-checker
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Prepare Training Data

- Place your `plagiarism_dataset.csv` in the project root.
- The CSV should have columns: `original`, `plagiarized`, `type`.

### 2. Run the Pipeline

```sh
python main.py
```

This will:

- Train the analysis engine on your dataset
- Create a dummy document corpus (for demo)
- Build and save a FAISS index
- Run two demo plagiarism searches (paraphrased and cross-lingual)

### 3. Results

- Results are printed to the console.
- You will see which documents are detected as sources, their scores, and detection methods.

---

## Project Structure

```
plag_checker/
│
├── engine/
│   ├── analysis_engine.py   # Model training & 1-to-1 analysis
│   ├── indexer.py           # FAISS index builder
│   └── search_engine.py     # Scalable search engine
│
├── utils/
│   └── text_utils.py        # Text similarity utilities
│
├── plagiarism_dataset.csv   # Your training data (required)
├── main.py                  # Main pipeline script
├── requirements.txt         # Python dependencies
└── ...
```

---

## Customization

- **Corpus:** Replace or extend the `create_dummy_corpus` function in `main.py` to use your own documents.
- **Training Data:** Use your own CSV for more robust detection.

---

## Notes

- The first run may take several minutes (model downloads, training, translation).
- For production, you can save and reload trained models to avoid retraining each time.

---

## License

MIT License

---

## Acknowledgements

- [Sentence Transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [XGBoost](https://xgboost.readthedocs.io/)
