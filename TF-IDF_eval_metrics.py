from collections import Counter, defaultdict
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def get_token_f1(pred, truth):
    p_toks, t_toks = pred.lower().split(), truth.lower().split()
    common = Counter(p_toks) & Counter(t_toks)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p_toks), same / len(t_toks)
    return 2 * prec * rec / (prec + rec)


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

pairs = [(acr, label) for acr, labels in lookup.items() for label in labels]
corpus = [f"{label} " + " ".join(docs[(acr, label)][:100]) for acr, label in pairs]
pair_to_idx = {pair: i for i, pair in enumerate(pairs)}

t_train_start = time.time()
vec = TfidfVectorizer(max_features=15000, stop_words="english")
matrix = vec.fit_transform(corpus)
train_time = time.time() - t_train_start

# 4. Filter test set to ONLY ambiguous abbreviations (2+ meanings)
ambig_test = [(acr, label, text) for acr, label, text in test if len(lookup[acr]) >= 2]
test_vecs = vec.transform([t[2] for t in ambig_test])

# 5. Evaluate Baseline vs. TF-IDF
truths = []
base_preds, base_top2, base_top3 = [], 0, 0
tfidf_preds, tfidf_top2, tfidf_top3 = [], 0, 0

t_base_start = time.time()
for acr, true_label, _ in ambig_test:
    candidates = list(lookup[acr])
    ranked_base = sorted(candidates, key=lambda c: counts[acr][c], reverse=True)
    base_preds.append(ranked_base[0])
    if true_label in ranked_base[:2]:
        base_top2 += 1
    if true_label in ranked_base[:3]:
        base_top3 += 1
base_ms = ((time.time() - t_base_start) / len(ambig_test)) * 1000

t_tfidf_start = time.time()
for i, (acr, true_label, _) in enumerate(ambig_test):
    truths.append(true_label)
    candidates = list(lookup[acr])
    row_ids = [pair_to_idx[(acr, c)] for c in candidates]
    scores = (test_vecs[i] @ matrix[row_ids].T).toarray()[0]
    ranked_tfidf = [candidates[idx] for idx in np.argsort(scores)[::-1]]

    tfidf_preds.append(ranked_tfidf[0])
    if true_label in ranked_tfidf[:2]:
        tfidf_top2 += 1
    if true_label in ranked_tfidf[:3]:
        tfidf_top3 += 1
tfidf_ms = ((time.time() - t_tfidf_start) / len(ambig_test)) * 1000


def print_metrics(name, preds, top2, top3, ms, t_time):
    total = len(truths)
    em = np.mean([p == t for p, t in zip(preds, truths)])
    macro_f1 = f1_score(truths, preds, average="macro", zero_division=0)
    tok_f1 = np.mean([get_token_f1(p, t) for p, t in zip(preds, truths)])

    print(f"\n=== {name} ===")
    print(f"Train Time:      {t_time:.2f} s")
    print(f"Latency:         {ms:.4f} ms/sample")
    print(f"Top-1 (Seen EM): {em:.2%}")
    print(f"Top-2 Accuracy:  {top2 / total:.2%}")
    print(f"Top-3 Accuracy:  {top3 / total:.2%}")
    print(f"Seen Macro F1:   {macro_f1:.2%}")
    print(f"Seen Token F1:   {tok_f1:.2%}")


print(f"Ambiguous samples tested: {len(truths)}")
print_metrics("MFS Baseline", base_preds, base_top2, base_top3, base_ms, 0.0)
print_metrics("TF-IDF Context Model", tfidf_preds, tfidf_top2, tfidf_top3, tfidf_ms, train_time)
