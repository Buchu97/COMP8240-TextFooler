# COMP8240 TextFooler

Replication experiments for TextFooler, the word-substitution attack introduced by Jin et al. (2020) in *Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment*.

The experiments use TextAttack 0.3.10. The attack ranks words by their effect on a classifier's prediction and searches for replacements subject to word-similarity, part-of-speech and sentence-similarity constraints.

## Implementation

- `N = 50`: maximum number of replacement candidates considered for each word.
- `delta = 0.7`: minimum word-embedding cosine similarity in the primary experiments, matching the value stated in the paper.
- `delta = 0.5`: TextAttack's default setting, retained for the BERT/MR threshold comparison.
- Full runs use 1,000 shuffled test examples with random seed `8240`.
- MR runs use the Hugging Face Rotten Tomatoes test split and TextAttack's released checkpoints. IMDB runs use the Hugging Face IMDB test split and TextAttack's released WordCNN checkpoint.

`scripts/textfooler_paper_delta_0_7.py` changes the word-embedding threshold in TextAttack's recipe from 0.5 to 0.7. The remaining recipe components are retained. The checkpoints, sampled test rows and software environment have not been established as identical to those used in the original paper.

## Completed experiments

All rows below contain 1,000 evaluated examples. The BERT/MR runs use the same examples, labels and initial predictions.

| Model | Dataset | delta | Original accuracy (%) | After-attack accuracy (%) | Attack success rate (%) | Perturbed words (%) | Mean queries | Mean USE cosine |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| WordCNN | MR | 0.7 | 77.0 | 1.8 | 97.66 | 18.28 | 46.74 | 0.8788 |
| WordLSTM | MR | 0.7 | 77.8 | 3.2 | 95.89 | 16.48 | 44.73 | 0.8893 |
| BERT | MR | 0.7 | 83.8 | 12.8 | 84.73 | 23.49 | 65.01 | 0.8465 |
| WordCNN | IMDB | 0.7 | 81.3 | 0.0 | 100.00 | 2.88 | 276.73 | 0.9842 |
| BERT | MR | 0.5 | 83.8 | 9.0 | 89.26 | 19.15 | 109.78 | 0.8379 |

Accuracy is calculated over all evaluated examples. Attack success rate is calculated over initially correct examples; initially incorrect examples are skipped. Perturbed-word percentages and USE-Large v3 cosine similarity are calculated over successful attacks. Mean query counts cover successful and failed attempts, excluding skipped examples.

The saved comparison with the paper's Table 3 is in [partial_table3_comparison.csv](outputs/main_table_full_PAPER_DELTA_0_7_20260930/partial_table3_comparison.csv). Four of the table's 15 model/dataset configurations have completed full runs. Twelve available configurations have completed one-example compatibility checks. The three Fake News configurations have no configured TextAttack checkpoint/dataset alias.

## Files

| Path | Contents |
|---|---|
| `scripts/` | Attack recipe, experiment matrix and semantic-similarity analysis |
| `tests/` | Tests for table coverage and attack-command construction |
| `outputs/main_table_full_PAPER_DELTA_0_7_20260930/` | WordCNN/MR, WordLSTM/MR and WordCNN/IMDB results; Table 3 comparison |
| `outputs/core_replication_PAPER_DELTA_0_7_bert_mr_1000/` | BERT/MR results with delta 0.7 |
| `outputs/core_replication_bert_mr_1000/` | BERT/MR results with delta 0.5 |
| `outputs/comparison_WORD_DELTA_0_5_vs_0_7/` | Controlled BERT/MR threshold comparison |
| `notes/` | Recorded software environment and hardware details |

Each full run includes attack examples in CSV and text format, a JSON attack summary, and a JSON semantic-similarity summary. Downloaded models, dataset caches, virtual environments and restart checkpoints are excluded from version control.

## Environment and commands

The recorded environment is Windows with Python 3.11.9, PyTorch 2.7.1 with CUDA 12.8, and an NVIDIA GeForce RTX 5060 Ti. Package versions are specified in `requirements-modern.txt`; the recorded dependency list is in `notes/environment-lock.txt`.

Setup and sentence-encoder download:

```powershell
.\setup.ps1
.\scripts\download_use_large_v3.ps1
```

BERT/MR with delta 0.7, writing to a new output directory:

```powershell
.\run_paper_delta_0_7_replication.ps1 -NumExamples 1000 -RunName bert_mr_delta_0_7_repeat
```

A selected Table 3 configuration:

```powershell
.\.venv\Scripts\python.exe scripts\main_table_matrix.py --num-examples 1000 --output-root outputs\table3_repeat --only wordcnn_mr
```

The matrix runner accepts model families `wordcnn`, `wordlstm` and `bert`, combined with dataset suffixes `mr`, `imdb`, `yelp` and `ag`. It saves a checkpoint every 50 examples for runs of at least 50 examples. Semantic analysis is a separate command for matrix runs:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_core_replication.py outputs\table3_repeat\wordcnn_mr\attack_results.csv outputs\table3_repeat\wordcnn_mr\semantic_similarity.json
```

Tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Sources

- Jin, D., Jin, Z., Zhou, J. T., and Szolovits, P. (2020). [Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment](https://ojs.aaai.org/index.php/AAAI/article/view/6311). AAAI, 34(05), 8018-8025.
- [Authors' TextFooler repository](https://github.com/jind11/TextFooler).
- Morris, J. X. et al. (2020). [TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP](https://aclanthology.org/2020.emnlp-demos.16/).
- [TextAttack repository](https://github.com/QData/TextAttack).
