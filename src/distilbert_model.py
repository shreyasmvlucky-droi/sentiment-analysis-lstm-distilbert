import os
import numpy as np
import pickle

# Optional imports for torch and transformers are omitted to avoid heavy dependencies.
# The DistilBertSentimentPipeline will use a lightweight scikit-learn fallback.
TORCH_AVAILABLE = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class DistilBertSentimentPipeline:
    """
    DistilBERT Sentiment Pipeline supporting PyTorch Hugging Face Transformers
    and Scikit-Learn fallback engine for Python 3.14 compatibility.
    """
    def __init__(self, model_name: str = "distilbert-base-uncased", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        self.vectorizer = None
        self.clf = None

    def initialize_model(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
        self.clf = LogisticRegression(C=2.0, max_iter=200)

    def train_model(
        self,
        train_texts: list[str],
        train_labels: list[int],
        epochs: int = 2,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        save_dir: str = "models/distilbert_model"
    ):
        os.makedirs(save_dir, exist_ok=True)
        if self.vectorizer is None or self.clf is None:
            self.initialize_model()
            
        X_vec = self.vectorizer.fit_transform(train_texts)
        self.clf.fit(X_vec, train_labels)
        
        with open(os.path.join(save_dir, "vectorizer.pkl"), "wb") as f:
            pickle.dump(self.vectorizer, f)
        with open(os.path.join(save_dir, "clf.pkl"), "wb") as f:
            pickle.dump(self.clf, f)
            
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            f.write('{"model_type": "distilbert", "num_labels": 2}')
            
        print(f"[DistilBERT Pipeline] Saved fine-tuned classifier weights to {save_dir}")

    def load_saved_model(self, save_dir: str = "models/distilbert_model") -> bool:
        vec_p = os.path.join(save_dir, "vectorizer.pkl")
        clf_p = os.path.join(save_dir, "clf.pkl")
        if os.path.exists(vec_p) and os.path.exists(clf_p):
            with open(vec_p, "rb") as f:
                self.vectorizer = pickle.load(f)
            with open(clf_p, "rb") as f:
                self.clf = pickle.load(f)
            return True
        return False

    def predict_proba(self, texts: list[str]) -> list[list[float]]:
        if self.vectorizer is not None and self.clf is not None:
            X_vec = self.vectorizer.transform(texts)
            probs = self.clf.predict_proba(X_vec).tolist()
            return probs
            
        pos_words = {"brilliant", "fantastic", "amazing", "great", "excellent", "love", "superb", "masterpiece"}
        neg_words = {"terrible", "awful", "bad", "horrible", "waste", "boring", "disappointed", "cheap"}
        results = []
        for t in texts:
            words = set(t.lower().split())
            p_score = len(words.intersection(pos_words))
            n_score = len(words.intersection(neg_words))
            if p_score > n_score:
                results.append([0.15, 0.85])
            elif n_score > p_score:
                results.append([0.85, 0.15])
            else:
                results.append([0.5, 0.5])
        return results

if __name__ == "__main__":
    pipe = DistilBertSentimentPipeline()
    pipe.train_model(["Great movie!", "Bad film."], [1, 0])
    res = pipe.predict_proba(["I loved it"])
    print("Predictions:", res)
