import nltk
from sklearn.metrics.pairwise import cosine_similarity

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

def char_ngram_jaccard(text1, text2, n=5):
    set1 = set([text1[i:i+n] for i in range(len(text1)-n+1)])
    set2 = set([text2[i:i+n] for i in range(len(text2)-n+1)])
    if not set1 or not set2:
        return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def normalized_levenshtein(text1, text2):
    distance = nltk.edit_distance(text1, text2)
    max_len = max(len(text1), len(text2))
    if max_len == 0:
        return 1.0
    return 1 - (distance / max_len)

def get_tfidf_score(vectorizer, text1, text2):
    try:
        return cosine_similarity(vectorizer.transform([text1]), vectorizer.transform([text2]))[0][0]
    except Exception:
        return 0.0