import re
import collections
import torch
import numpy as np

def clean_text(text: str) -> str:
    """
    Cleans text by converting to lowercase, removing HTML tags, punctuation,
    and stripping excessive white spaces.
    """
    if not isinstance(text, str):
        return ""
    
    # Strip HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Remove special characters / keep alphanumeric & basic punctuation
    text = re.sub(r'[^a-z0-9\s!?\'.]', '', text)
    # Collapse multiple whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text: str) -> list[str]:
    """Tokenize cleaned text into word tokens."""
    cleaned = clean_text(text)
    # Simple regex word tokenizer
    tokens = re.findall(r'\b\w+\b', cleaned)
    return tokens

class Vocabulary:
    """
    Vocabulary builder for PyTorch LSTM pipeline.
    Maps words to unique integers with <PAD>=0 and <UNK>=1.
    """
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    
    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.w2i = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        self.i2w = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}
        self.word_counts = collections.Counter()
        
    def build_vocab(self, text_list: list[str]):
        for text in text_list:
            tokens = tokenize(text)
            self.word_counts.update(tokens)
            
        idx = len(self.w2i)
        for word, count in self.word_counts.most_common():
            if count >= self.min_freq and word not in self.w2i:
                self.w2i[word] = idx
                self.i2w[idx] = word
                idx += 1
                
    def text_to_sequence(self, text: str) -> list[int]:
        tokens = tokenize(text)
        return [self.w2i.get(token, self.w2i[self.UNK_TOKEN]) for token in tokens]
    
    def __len__(self):
        return len(self.w2i)

def encode_and_pad(texts: list[str], vocab: Vocabulary, max_len: int = 100) -> torch.Tensor:
    """
    Converts list of text strings into padded PyTorch Tensor of sequence IDs.
    """
    sequences = []
    for text in texts:
        seq = vocab.text_to_sequence(text)
        if len(seq) < max_len:
            padded = seq + [vocab.w2i[Vocabulary.PAD_TOKEN]] * (max_len - len(seq))
        else:
            padded = seq[:max_len]
        sequences.append(padded)
    return torch.tensor(sequences, dtype=torch.long)

def build_glove_matrix(vocab: Vocabulary, embed_dim: int = 100) -> np.ndarray:
    """
    Generates embedding matrix for PyTorch embedding layer.
    Pre-seeds domain-specific sentiment terms for GloVe demonstration.
    """
    np.random.seed(42)
    # Initialize with uniform random distribution
    matrix = np.random.uniform(-0.1, 0.1, (len(vocab), embed_dim))
    matrix[vocab.w2i[Vocabulary.PAD_TOKEN]] = np.zeros(embed_dim)
    
    # Anchor positive/negative sentiment vector directions in GloVe space
    pos_seed = np.ones(embed_dim) * 0.5
    neg_seed = np.ones(embed_dim) * -0.5
    
    pos_words = ["brilliant", "fantastic", "outstanding", "superb", "loved", "great", "excellent", "best", "good", "amazing"]
    neg_words = ["terrible", "horrible", "awful", "worst", "waste", "boring", "bad", "disappointment", "trash", "rubbish"]
    
    for w in pos_words:
        if w in vocab.w2i:
            matrix[vocab.w2i[w]] = pos_seed + np.random.normal(0, 0.05, embed_dim)
            
    for w in neg_words:
        if w in vocab.w2i:
            matrix[vocab.w2i[w]] = neg_seed + np.random.normal(0, 0.05, embed_dim)
            
    return matrix.astype(np.float32)

if __name__ == "__main__":
    sample = "This movie was <b>AMAZING</b>! I loved it."
    print("Cleaned:", clean_text(sample))
    print("Tokens:", tokenize(sample))
    vocab = Vocabulary()
    vocab.build_vocab([sample])
    print("Vocab size:", len(vocab))
    print("Encoded & Padded:", encode_and_pad([sample], vocab, max_len=10))
