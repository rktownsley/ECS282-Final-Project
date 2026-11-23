# Open-Vocabulary Medical Abbreviation Disambiguation

This project evaluates whether a small, locally runnable language model can expand ambiguous medical abbreviations from clinical and biomedical context. It compares prompted and parameter-efficient generative models with closed-vocabulary BERT and lexical-retrieval baselines.

The primary generative model is `unsloth/Llama-3.2-1B-Instruct`, adapted with 4-bit QLoRA. The experiments use deterministic decoding and require the model to output only the canonical expansion string.

## Repository contents

- `01_baseline_llama3_2_1b_prompting.ipynb` — zero-shot and three-shot Llama baselines.
- `02_baseline_bioclinicalbert_closed_vocab.ipynb` — BioClinicalBERT classifier baseline.
- `03_baseline_pubmedbert_closed_vocab.ipynb` — PubMedBERT classifier baseline.
- `04_main_llama3_2_1b_qlora_r16_greedy.ipynb` — primary QLoRA experiment with rank 16 and greedy decoding.
- `05a_ablation_llama3_2_1b_qlora_r8.ipynb` — rank ablation using rank-8 QLoRA with local context and greedy decoding.
- `05b_ablation_llama3_2_1b_neftune.ipynb` — regularization ablation using rank-16 QLoRA with NEFTune.
- `05_ablation_llama3_2_1b_qlora_r8_neftune_leadctx.ipynb` — combined ablation with rank-8 QLoRA, NEFTune, lead-plus-local context, and beam decoding.
- `06_improve_unseen_diverse_reg_fullctx.ipynb` — unseen-acronym experiments, each behind a flag: training on 1,000 examples spread across ~275 acronyms (`DIVERSE_TRAIN`), regularization with early stopping on held-out dev acronyms (`REGULARIZED`), and whole-abstract context (`CONTEXT_MODE`). Uses the same seen/unseen test sets as 01–05.
- `TF-IDF_eval_metrics.py` — TF–IDF retrieval evaluation with top-1, top-2, top-3, and F1 metrics.
- `TF-IDF_and_MFS_baseline.py` — compact TF–IDF and most-frequent-sense baseline script.
- `results-2.txt` — detailed benchmark results, including latency and peak VRAM.
- `proposal.txt` — project proposal and methodology description.

## Dataset

Download `valid.csv` from the [MeDAL dataset](https://zenodo.org/records/4482922) and place it in Google Drive. The notebooks expect:

```text
/content/drive/MyDrive/valid.csv
```

The neural-model notebooks read the first 300,000 rows, remove invalid records, retain genuinely ambiguous acronyms, and keep labels with at least 30 examples. The 40 most frequent ambiguous acronyms are split into 30 seen and 10 unseen acronyms using seed `42`. The shared in-memory benchmark contains:

- 1,000 training examples
- 500 seen-acronym test examples
- 500 unseen-acronym test examples

The TF–IDF/MFS evaluation uses the full `valid.csv`, creates an 80/20 sentence split, and evaluates on 198,075 ambiguous test examples.

## Running the experiments

The notebooks are designed for Google Colab with a GPU runtime; an NVIDIA T4 with 16 GB VRAM is the expected environment.

1. Open a notebook in Google Colab.
2. Select a GPU runtime.
3. Upload `valid.csv` to the Google Drive location above.
4. Run the notebooks from top to bottom.

Each notebook installs its own dependencies, including `transformers`, `datasets`, `accelerate`, `peft`, `bitsandbytes`, and `scikit-learn` where needed. Run the notebooks in numeric order to reproduce the neural-model progression. Model downloads require an internet connection, and Hugging Face authentication may be needed if a selected model repository requires it.

The standalone TF–IDF scripts can be run locally from the repository after placing `valid.csv` in the project directory:

```bash
python -m pip install numpy pandas scikit-learn
python TF-IDF_eval_metrics.py
```

Use `TF-IDF_and_MFS_baseline.py` for the shorter top-1 comparison. The neural notebooks should be run in Colab because they require GPU-backed model loading and fine-tuning.

## Method

Each neural-model example marks the target abbreviation in context and asks the model to return only its expansion. The primary QLoRA run uses 4-bit NF4 quantization, rank-16 adapters, a learning rate of `2e-4`, three epochs, and an effective batch size of 16. The ablations separately test rank reduction (`05a`), NEFTune regularization (`05b`), and the combined rank/context/beam configuration (`05`). Evaluation reports exact match, macro F1, and token-level F1 separately for seen and unseen abbreviations.

## Results

All neural-model rows below use the same `1,000 / 500 / 500` split, seed `42`, and T4 runtime. See [`results-2.txt`](results-2.txt) for the complete tables.

| Model | Seen EM | Seen Macro F1 | Unseen EM | Unseen Token F1 |
| --- | ---: | ---: | ---: | ---: |
| Llama zero-shot | 0.60% | 0.27% | 1.20% | 4.70% |
| Llama three-shot | 1.40% | 0.56% | 1.60% | 5.98% |
| BioClinicalBERT | 37.00% | 31.96% | 0.00% | 0.53% |
| PubMedBERT | 54.40% | 48.41% | 0.00% | 0.00% |
| Llama + QLoRA, rank 16 | **76.00%** | **61.80%** | 8.80% | 14.13% |
| Llama + QLoRA, rank 8 (`05a`) | 74.80% | 60.49% | 8.20% | 14.01% |
| Llama + QLoRA + NEFTune (`05b`) | 74.00% | 60.45% | 10.00% | 16.17% |
| Llama + QLoRA, rank 8 + NEFTune + context/beam (`05`) | 70.80% | 53.53% | **12.00%** | **18.10%** |

The separate lexical benchmark reports 30.17% top-1 accuracy for the most-frequent-sense baseline and 79.71% top-1 accuracy for TF–IDF on its larger ambiguous test set. These scores are not directly comparable to the neural benchmark because the data split and evaluation set differ.

Overall, QLoRA substantially improves in-domain abbreviation expansion. The rank-8 and NEFTune ablations provide modest unseen-acronym gains, while the combined context-and-beam configuration achieves the strongest unseen performance at the cost of lower seen performance and higher inference latency.

## Reproducibility notes

**Prompt-construction fix (01, 04, 05, 05a, 05b).** Earlier versions truncated the tokenized prompt from the right and added a second `<|begin_of_text|>` at evaluation time. Measured on MeDAL abstracts with the Llama 3.2 tokenizer, the cut removed the `Expansion:` line and the assistant header from about 34% of training prompts in 04/05a/05b (224-token budget), about 12% in 05, and about 46% of three-shot prompts in 01. The notebooks now leave room for full prompts, narrow the word window if a prompt is still too long, and never add special tokens on top of the chat template. The result tables above come from the earlier versions and should be regenerated.

Results depend on the exact dataset file, the rows read by the notebooks, seed `42`, model revisions, and the available GPU. Neural notebooks keep temporary checkpoints in the Colab runtime and do not write checkpoints to Google Drive.

## Authors

Riyan Townsley and Sawyer Blankenship, Department of Computer Science, University of California, Davis.
