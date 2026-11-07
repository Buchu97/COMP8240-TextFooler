# COMP8240 TextFooler replication

This project reproduces the main classification and BERT entailment experiments in [Jin et al. (2020)](https://ojs.aaai.org/index.php/AAAI/article/view/6311) and tests whether the attack generalises to new text-classification tasks. The runs use TextAttack 0.3.10 with up to 50 replacement candidates and a minimum word-embedding cosine similarity of 0.7. The paired ablations remove only the sentence-similarity constraint.

## Setup

The recorded attack environment uses Windows, Python 3.11, PyTorch 2.7.1 with CUDA 12.8, and an NVIDIA RTX 5060 Ti. Run the following in PowerShell from the repository root. Setup requires the Python 3.11 launcher, Git and internet access.

```powershell
.\setup.ps1
$python = '.\.venv\Scripts\python.exe'
& $python -m unittest discover -s tests
```

## Original-data replication

Prepare the attack samples from the authors' pinned release and download the BERT/MR checkpoint. A five-example smoke run uses the same model and recipe as the full MR run.

```powershell
git clone -c core.autocrlf=false https://github.com/jind11/TextFooler.git repo/TextFooler-official
git -C repo/TextFooler-official checkout --detach 6a56cb10f1055e00a8fbc6882f289e91cfe60f4e
& $python scripts/download_original_models.py --models bert_mr
& $python scripts/replication_runs.py prepare --dataset mr --num-examples 5 --output-dir data/processed/smoke_mr
& $python scripts/replication_runs.py run --sample-dir data/processed/smoke_mr --output-dir outputs/smoke_mr --model original-bert-mr
```

For the full run, prepare all eight released partitions and fetch the checkpoints from the [catalog](notes/original_checkpoint_catalog.json). Use a new output directory for each independent run. The sentence encoder used during the attack is fetched on first use. USE-Large v3 is a separate encoder used for post-hoc comparison.

```powershell
foreach ($dataset in 'mr','imdb','yelp','ag','fake','snli','mnli_m','mnli_mm') {
    & $python scripts/replication_runs.py prepare --dataset $dataset --output-dir "data/processed/released/$dataset"
}
$catalog = Get-Content -Raw notes/original_checkpoint_catalog.json | ConvertFrom-Json
$modelNames = @($catalog.PSObject.Properties.Name)
& $python scripts/download_original_models.py --models $modelNames
& $python scripts/run_replication_suite.py --samples data/processed/released --output-root outputs/replication --workers 1
.\scripts\download_use_large_v3.ps1
& $python scripts/replication_report.py --root outputs/replication --report-dir outputs/replication_report --similarity
& $python scripts/verify_replication_artifacts.py --root outputs/replication --samples data/processed/released --output outputs/replication_report/verification.json
```

The suite covers 15 classification runs, three BERT entailment runs and two BERT ablations. The authors' [checkpoint note](https://github.com/jind11/TextFooler/issues/47#issuecomment-955085106) explains the missing ESIM and InferSent settings. Transfer attacks and adversarial training are outside this replication.

## Sources

- [TextFooler paper and authors' release](https://github.com/jind11/TextFooler)
- [TextAttack paper](https://aclanthology.org/2020.emnlp-demos.16/) and [TextAttack source](https://github.com/QData/TextAttack)
- Upstream notices for the released TextFooler preprocessing and TextAttack adaptations are in [licenses/](licenses/).
