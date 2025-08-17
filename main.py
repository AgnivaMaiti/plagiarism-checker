import os

# Import our new package structure
from engine.analysis_engine import PlagiarismAnalyzer
from engine.indexer import Indexer
from engine.search_engine import SearchEngine


def create_dummy_corpus(path="corpus", num_files=20):
    """Creates a folder with sample text files to act as our searchable database."""
    print(f"\nCreating a dummy document corpus at '{path}'...")
    os.makedirs(path, exist_ok=True)
    sample_texts = [
        "The Apollo program was the third United States human spaceflight program carried out by NASA.",
        "Quantum mechanics is a fundamental theory in physics that provides a description of the physical properties of nature at the scale of atoms and subatomic particles.",
        "The Renaissance was a period in European history marking the transition from the Middle Ages to modernity.",
        "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy.",
        "Machine learning is a field of study in artificial intelligence concerned with the development and study of statistical algorithms.",
        "Beethoven's 5th Symphony is one of the most famous compositions in classical music.",
        "The novel 'To Kill a Mockingbird' was written by Harper Lee and published in 1960.",
        "Artificial intelligence is a branch of computer science that is crucial for modern technology."
    ]
    for i in range(num_files):
        doc_id = f"doc_{i+1}.txt"
        content = sample_texts[i % len(sample_texts)] + f" (This is content from document {i+1})"
        with open(os.path.join(path, doc_id), "w", encoding='utf-8') as f:
            f.write(content)
    print(f"✅ Created {num_files} sample documents.")


def main():
    """Main function to run the complete plagiarism detection pipeline."""
    
    # --- Step 1: Initialize and Train the Core Analysis Engine ---
    # In a real application, you would train this once and save the models.
    # For this demo, we train it every time.
    print("--- Initializing and Training the Analysis Engine ---")
    analyzer = PlagiarismAnalyzer()
    # IMPORTANT: Make sure 'plagiarism_dataset.csv' is in the same directory as this script.
    try:
        analyzer.train(training_data_path='plagiarism_dataset.csv')
    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: {e}")
        print("Please download your 'plagiarism_dataset.csv' and place it in the root of the 'plag_checker' directory.")
        return

    # --- Step 2: Create a Corpus and Index it ---
    CORPUS_PATH = "corpus"
    create_dummy_corpus(path=CORPUS_PATH)
    
    indexer = Indexer(sbert_model=analyzer.sbert_model)
    indexer.build_index(corpus_path=CORPUS_PATH)
    indexer.save()

    # --- Step 3: Load the Index and Perform Searches ---
    search_engine = SearchEngine(analysis_engine=analyzer)
    if search_engine.load_index(corpus_path=CORPUS_PATH):
        
        print("\n" + "="*80)
        print("                     --- SCALABLE SEARCH DEMONSTRATION ---")
        print("="*80)

        # --- Query 1: A paraphrased version of a known document ---
        query1 = "The study and creation of statistical algorithms is a subfield of AI known as machine learning."
        print(f"\nCASE 1: Searching with a paraphrased query...")
        print(f"Query Text: '{query1}'")
        sources = search_engine.find_plagiarism_sources(query1)
        if sources:
            print("\n--- Plagiarism Found! ---")
            for source in sources:
                print(f"  - Source: {source['source_document']}, Score: {source['plagiarism_score']}")
                print(f"    Detection Method: {source['details']['detection_method']}")
        else:
            print("\n--- No significant plagiarism detected. ---")

        # --- Query 2: A cross-lingual version of a known document ---
        query2 = "Künstliche Intelligenz ist ein Zweig der Informatik, der für die moderne Technologie entscheidend ist." # German
        print(f"\nCASE 2: Searching with a cross-lingual query...")
        print(f"Query Text: '{query2}'")
        sources = search_engine.find_plagiarism_sources(query2)
        if sources:
            print("\n--- Plagiarism Found! ---")
            for source in sources:
                print(f"  - Source: {source['source_document']}, Score: {source['plagiarism_score']}")
                print(f"    Detection Method: {source['details']['detection_method']}")
        else:
            print("\n--- No significant plagiarism detected. ---")

if __name__ == "__main__":
    main()