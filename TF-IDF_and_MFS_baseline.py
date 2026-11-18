from collections import Counter, defaultdict
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# 1. Load data
df = pd.read_csv("valid.csv")
records = []
for text, loc, label in zip(df["TEXT"], df["LOCATION"], df["LABEL"]):
    words = str(text).lower().split()
    if 0 <= int(loc) < len(words):
        acr = words[int(loc)].strip(".,;:()[]{}\"'")
        records.append((acr, str(label).strip(), str(text).lower()))

# 2. Build 100% lookup table from all rows
lookup = defaultdict(set)
for acr, label, _ in records:
    lookup[acr].add(label)

# 3. Split 80/20 to train context profiles and test on unseen sentences
train, test = train_test_split(records, test_size=0.2, random_state=42)

docs = defaultdict(list)
counts = defaultdict(Counter)
for acr, label, text in train:
    counts[acr][label] += 1
    docs[(acr, label)].append(text)

# One context document per (abbreviation, full_term)
pairs = [(acr, label) for acr, labels in lookup.items() for label in labels]
corpus = [f"{label} " + " ".join(docs[(acr, label)][:100]) for acr, label in pairs]
pair_to_idx = {pair: i for i, pair in enumerate(pairs)}

vec = TfidfVectorizer(max_features=15000, stop_words="english")
matrix = vec.fit_transform(corpus)

# 4. Filter test set to ONLY ambiguous abbreviations (2+ meanings)
ambig_test = [(acr, label, text) for acr, label, text in test if len(lookup[acr]) >= 2]
test_vecs = vec.transform([t[2] for t in ambig_test])

# 5. Evaluate Baseline vs. TF-IDF
base_hits = 0
tfidf_hits = 0

for i, (acr, true_label, _) in enumerate(ambig_test):
    candidates = list(lookup[acr])

    # Baseline: most frequent in training set
    base_pred = max(candidates, key=lambda c: counts[acr][c])
    if base_pred == true_label:
        base_hits += 1

    # TF-IDF: highest cosine similarity to context
    row_ids = [pair_to_idx[(acr, c)] for c in candidates]
    scores = (test_vecs[i] @ matrix[row_ids].T).toarray()[0]
    tfidf_pred = candidates[scores.argmax()]
    if tfidf_pred == true_label:
        tfidf_hits += 1

total = len(ambig_test)
print(f"Ambiguous samples tested: {total}")
print(f"Baseline (Most Frequent): {base_hits / total:.2%}")
print(f"TF-IDF Context Model:     {tfidf_hits / total:.2%}")
