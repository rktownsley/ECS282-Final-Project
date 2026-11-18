# Open-Vocabulary Medical Abbreviation Disambiguation

This project evaluates whether a small, locally runnable language model can expand ambiguous medical abbreviations from clinical and biomedical context. It compares prompted and parameter-efficient generative models with closed-vocabulary BERT baselines.

The main generative model is `unsloth/Llama-3.2-1B-Instruct`, adapted with 4-bit QLoRA. The experiments use deterministic decoding and require the model to output only the canonical expansion string.

## Repository contents

- `01_baseline_llama3_2_1b_prompting.ipynb` — zero-shot and three-shot Llama baselines.
- `02_baseline_bioclinicalbert_closed_vocab.ipynb` — BioClinicalBERT classifier baseline.
- `03_baseline_pubmedbert_closed_vocab.ipynb` — PubMedBERT classifier baseline.
- `04_main_llama3_2_1b_qlora_r16_greedy.ipynb` — primary QLoRA experiment with greedy decoding.
- `05_ablation_llama3_2_1b_qlora_r8_neftune_leadctx.ipynb` — QLoRA ablation using a different rank, context, NEFTune, and beam decoding.
- `results.txt` — consolidated benchmark table.
- `proposal.txt` — project proposal and methodology description.

## Dataset

Download `valid.csv` from the [MeDAL dataset](https://zenodo.org/records/4482922) and place it in Google Drive. The notebooks expect it at:

```text
/content/drive/MyDrive/valid.csv
```

The notebooks read the first 300,000 rows, retain valid `TEXT`, `LOCATION`, and `LABEL` records, and filter for genuinely ambiguous acronyms. With seed `42`, the final benchmark uses:

- 1,000 training examples
- 500 seen-acronym test examples
- 500 unseen-acronym test examples

The same in-memory split is used for the baseline and fine-tuned comparisons.

## Running the experiments

These notebooks are designed for Google Colab with a GPU runtime. A single NVIDIA T4 with 16 GB VRAM is the expected environment.

1. Open a notebook in Google Colab.
2. Select a GPU runtime.
3. Upload `valid.csv` to the Google Drive location above.
4. Run the notebook from top to bottom.

Each notebook installs its own Python dependencies, including `transformers`, `datasets`, `accelerate`, `peft`, `bitsandbytes`, and `scikit-learn` where needed. Run the notebooks in numeric order to reproduce the progression from prompting baselines through QLoRA experiments. Model downloads require an internet connection; Hugging Face authentication may be needed if the selected model repository requests it.

## Method

For each example, the target abbreviation is marked in context and the model is prompted to return only its expansion. The main QLoRA run uses 4-bit NF4 quantization, rank-16 adapters, a learning rate of `2e-4`, three epochs, and an effective batch size of 16. Evaluation reports exact match, macro F1, and token-level F1 separately for seen and unseen abbreviations.

## Reported benchmark

The consolidated results in `results.txt` were measured on the same split using an NVIDIA T4:

| Model | Seen EM | Seen Macro F1 | Unseen EM | Unseen Token F1 |
| --- | ---: | ---: | ---: | ---: |
| Llama zero-shot | 0.60% | 0.27% | 1.20% | 4.70% |
| Llama three-shot | 1.40% | 0.56% | 1.60% | 5.98% |
| BioClinicalBERT | 37.00% | 31.96% | 0.00% | 0.53% |
| PubMedBERT | 54.40% | 48.41% | 0.00% | 0.00% |
| Llama + QLoRA, rank 16 | **76.00%** | **61.80%** | 8.80% | 14.13% |
| Llama + QLoRA, rank 8 ablation | 70.80% | 53.53% | **12.00%** | **18.10%** |

The results suggest that QLoRA substantially improves in-domain abbreviation expansion, while performance on acronyms excluded from training remains considerably harder.

## Reproducibility notes

Results depend on the exact dataset file, the first 300,000 rows read by the notebooks, random seed `42`, model revisions, and the available GPU. The notebooks keep temporary checkpoints in the Colab runtime and do not write checkpoints to Google Drive.

## Authors

Riyan Townsley and Sawyer Blankenship, Department of Computer Science, University of California, Davis.
