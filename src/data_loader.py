import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def generate_benchmark_dataset(num_samples: int = 1500, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a rich, realistic benchmark Sentiment Dataset containing diverse 
    movie & product reviews for binary sentiment classification (0: Negative, 1: Positive).
    """
    np.random.seed(random_state)
    
    pos_templates = [
        "This movie was absolutely {adj}! The acting by the lead cast was {adj2} and story was thrilling.",
        "An astonishing masterpiece of contemporary cinema. The direction was {adj} and visuals were {adj2}.",
        "I loved every single minute of this film! Truly {adj} storyline with {adj2} execution.",
        "Sensational performance and brilliant screenplay. Extremely {adj} and {adj2} experience.",
        "Highly recommended! A breathtaking watch with {adj} emotion and {adj2} pacing.",
        "The product exceeded all my expectations. Superb build quality and {adj} performance.",
        "Outstanding customer service and remarkably fast delivery. Totally {adj} value for money.",
        "Exceptional quality, sleek design, and unbelievably {adj} user experience. Would buy again!",
        "Flawless execution and impressive depth. A {adj} victory for modern storytelling.",
        "Enchanting score, top-tier cinematography, and an utterly {adj} performance."
    ]
    
    neg_templates = [
        "What a total waste of time and money! The plot was {adj_n} and acting was completely {adj_n2}.",
        "Terrible experience. The script was absurdly {adj_n}, predictable, and painfully {adj_n2}.",
        "I deeply regret watching this. The direction was {adj_n} and characters were totally {adj_n2}.",
        "Dreadful film with awful dialogue. Horribly {adj_n} pacing and overall a huge disappointment.",
        "Cheap quality and broke after two days of use. Extremely {adj_n} and {adj_n2}.",
        "Unacceptable quality control. The product felt {adj_n} and performance was miserably {adj_n2}.",
        "Frustrating user interface, constant crashes, and completely {adj_n} customer support.",
        "Boring, uninspired, and ridiculously {adj_n}. Do not waste your money on this garbage.",
        "Disappointing from start to finish. Tedious narrative and {adj_n} acting choices.",
        "Utter disaster of a project. Clunky mechanics, {adj_n} visuals, and atrocious sound design."
    ]
    
    pos_adj = ["brilliant", "fantastic", "outstanding", "spectacular", "superb", "phenomenal", "masterful", "wonderful", "captivating", "stunning"]
    pos_adj2 = ["inspiring", "heartwarming", "flawless", "unforgettable", "delightful", "top-notch", "compelling", "magical"]
    
    neg_adj = ["horrible", "atrocious", "disastrous", "dreadful", "abysmal", "pathetic", "clunky", "shallow", "monotonous", "ridiculous"]
    neg_adj2 = ["irritating", "pointless", "unbearable", "worthless", "frustrating", "flawed", "flat", "painful"]
    
    records = []
    
    for _ in range(num_samples // 2):
        tpl = np.random.choice(pos_templates)
        text = tpl.format(adj=np.random.choice(pos_adj), adj2=np.random.choice(pos_adj2))
        records.append({"text": text, "label": 1})
        
        tpl_n = np.random.choice(neg_templates)
        text_n = tpl_n.format(adj_n=np.random.choice(neg_adj), adj_n2=np.random.choice(neg_adj2))
        records.append({"text": text_n, "label": 0})
        
    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df

def load_sentiment_dataset(data_dir: str = "data", test_size: float = 0.2, random_state: int = 42):
    """
    Loads sentiment dataset from file if exists, or generates benchmark dataset.
    Returns train_df and test_df.
    """
    raw_path = os.path.join(data_dir, "raw", "sentiment_data.csv")
    
    if os.path.exists(raw_path):
        df = pd.read_csv(raw_path)
        print(f"[DataLoader] Loaded dataset from {raw_path} ({len(df)} samples)")
    else:
        print("[DataLoader] Raw file not found. Generating high-quality benchmark dataset...")
        df = generate_benchmark_dataset(num_samples=2000, random_state=random_state)
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        df.to_csv(raw_path, index=False)
        print(f"[DataLoader] Saved generated benchmark dataset to {raw_path}")
        
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['label'])
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

if __name__ == "__main__":
    train, test = load_sentiment_dataset()
    print(f"Train samples: {len(train)}, Test samples: {len(test)}")
    print("Sample positive review:", train[train['label'] == 1].iloc[0]['text'])
    print("Sample negative review:", train[train['label'] == 0].iloc[0]['text'])
