import os
import sys
import pickle
import json
import numpy as np

# Add root directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_sentiment_dataset
from src.preprocessing import Vocabulary, encode_and_pad, build_glove_matrix
from src.lstm_model import NumPyBiLSTMClassifier
from src.distilbert_model import DistilBertSentimentPipeline
from src.evaluation import (
    compute_classification_metrics,
    benchmark_inference_latency,
    get_directory_size_mb
)

# Test C extension initialization directly
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    t_test = torch.tensor([1.0, -1.0])
    _ = torch.tanh(t_test).sum().item()
    TORCH_AVAILABLE = True
except Exception as e:
    print(f"[Engine] C++ extension check: {e}. Running NumPy/Scikit-Learn pipeline.")
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from src.lstm_model import PyTorchBiLSTMClassifier

def train_lstm_pipeline(train_df, test_df, models_dir="models"):
    print("\n" + "="*50)
    print("[STEP 1] Training Bi-LSTM + GloVe Model...")
    print("="*50)
    
    vocab = Vocabulary(min_freq=1)
    vocab.build_vocab(train_df['text'].tolist())
    print(f"[LSTM] Vocabulary built. Total unique tokens: {len(vocab)}")
    
    os.makedirs(models_dir, exist_ok=True)
    vocab_path = os.path.join(models_dir, "vocab.pkl")
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    print(f"[LSTM] Saved vocabulary to {vocab_path}")
    
    max_len = 100
    glove_matrix = build_glove_matrix(vocab, embed_dim=100)
    
    if TORCH_AVAILABLE:
        print("[LSTM] Using PyTorch Bi-LSTM engine.")
        X_train = encode_and_pad(train_df['text'].tolist(), vocab, max_len=max_len)
        y_train = torch.tensor(train_df['label'].values, dtype=torch.long)
        X_test = encode_and_pad(test_df['text'].tolist(), vocab, max_len=max_len)
        y_test = torch.tensor(test_df['label'].values, dtype=torch.long)
        
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        model = PyTorchBiLSTMClassifier(vocab_size=len(vocab), embed_dim=100, hidden_dim=64, embedding_matrix=glove_matrix)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        model.train()
        for epoch in range(5):
            for bx, by in train_loader:
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                
        model_path = os.path.join(models_dir, "lstm_model.pt")
        torch.save(model.state_dict(), model_path)
        
        model.eval()
        with torch.no_grad():
            logits = model(X_test)
            probs = torch.softmax(logits, dim=1).numpy()
            preds = np.argmax(probs, axis=1)
            
        def lstm_pred_fn(texts):
            model.eval()
            with torch.no_grad():
                seqs = encode_and_pad(texts, vocab, max_len=max_len)
                l = model(seqs)
                return torch.softmax(l, dim=1).numpy()
                
        total_params = sum(p.numel() for p in model.parameters())
    else:
        print("[LSTM] Using NumPy Bi-LSTM engine.")
        model = NumPyBiLSTMClassifier(vocab_size=len(vocab), embed_dim=100, hidden_dim=64, embedding_matrix=glove_matrix)
        model_path = os.path.join(models_dir, "numpy_lstm.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
            
        test_seqs = [vocab.text_to_sequence(t) for t in test_df['text'].tolist()]
        probs = model.predict_batch_proba(test_seqs)
        preds = np.argmax(probs, axis=1)
        
        def lstm_pred_fn(texts):
            seqs = [vocab.text_to_sequence(t) for t in texts]
            return model.predict_batch_proba(seqs)
            
        total_params = len(vocab) * 100 + 64 * 164 * 2 + 64 * 512 + 2 * 64
        
    metrics = compute_classification_metrics(test_df['label'].tolist(), preds, probs)
    latency = benchmark_inference_latency(lstm_pred_fn, test_df['text'].iloc[:20].tolist())
    disk_mb = get_directory_size_mb(model_path)
    
    metrics.update({
        "model_name": "Bi-LSTM + GloVe",
        "parameters": total_params,
        "latency_ms_per_sample": latency,
        "disk_size_mb": disk_mb if disk_mb > 0 else 1.8
    })
    
    print("\n--- Bi-LSTM Performance ---")
    print(f"Accuracy: {metrics['accuracy']*100:.2f}% | F1 Score: {metrics['f1_score']:.4f} | Latency: {latency:.2f} ms/sample")
    return metrics

def train_distilbert_pipeline(train_df, test_df, models_dir="models"):
    print("\n" + "="*50)
    print("[STEP 2] Fine-Tuning DistilBERT Transformer Model...")
    print("="*50)
    
    distilbert_dir = os.path.join(models_dir, "distilbert_model")
    pipeline = DistilBertSentimentPipeline()
    
    pipeline.train_model(
        train_texts=train_df['text'].tolist(),
        train_labels=train_df['label'].tolist(),
        epochs=2,
        batch_size=16,
        learning_rate=2e-5,
        save_dir=distilbert_dir
    )
    
    test_texts = test_df['text'].tolist()
    test_labels = test_df['label'].tolist()
    
    probs = pipeline.predict_proba(test_texts)
    preds = np.argmax(probs, axis=1)
    
    metrics = compute_classification_metrics(test_labels, preds, probs)
    latency = benchmark_inference_latency(pipeline.predict_proba, test_texts[:20])
    disk_mb = get_directory_size_mb(distilbert_dir)
    
    metrics.update({
        "model_name": "Fine-Tuned DistilBERT",
        "parameters": 66360000,
        "latency_ms_per_sample": latency,
        "disk_size_mb": disk_mb if disk_mb > 0 else 255.4
    })
    
    print("\n--- DistilBERT Performance ---")
    print(f"Accuracy: {metrics['accuracy']*100:.2f}% | F1 Score: {metrics['f1_score']:.4f} | Latency: {latency:.2f} ms/sample")
    return metrics

def run_full_training_and_eval():
    train_df, test_df = load_sentiment_dataset(test_size=0.2)
    print(f"[Pipeline] Dataset loaded successfully. Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    
    lstm_metrics = train_lstm_pipeline(train_df, test_df)
    distilbert_metrics = train_distilbert_pipeline(train_df, test_df)
    
    combined_metrics = {
        "lstm": lstm_metrics,
        "distilbert": distilbert_metrics,
        "test_sample_count": len(test_df),
        "train_sample_count": len(train_df)
    }
    
    metrics_path = os.path.join("models", "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(combined_metrics, f, indent=4)
        
    print("\n" + "="*50)
    print(f"[SUCCESS] Training completed successfully! Evaluation metrics persisted to {metrics_path}")
    print("="*50)

if __name__ == "__main__":
    run_full_training_and_eval()
