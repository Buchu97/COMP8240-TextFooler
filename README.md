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

## New-data evaluation

### Existing datasets

The Type 1 experiments use the published Poem Sentiment, CLINC150, News Category, SciCite and Environmental Claims datasets. The preparation scripts pin sources, record label mappings and build fixed evaluation sets. Poem Sentiment and Environmental Claims use their full test splits. The other three use 1,000 held-out examples per dataset. News uses a locally trained DistilBERT model and SciCite uses a locally trained SciBERT model. The other runs use published classifiers.

```powershell
& $python scripts/prepare_type1.py --output-root data/processed/type1
& $python scripts/prepare_type1_additions.py --output-root data/processed/type1_additions
& $python scripts/prepare_scibert_base.py
& $python scripts/train_news.py --dataset-dir data/processed/type1/news --model-dir models/news --report-dir outputs/news_training
& $python scripts/train_news.py --dataset-dir data/processed/type1_additions/scicite --model-dir models/scicite --report-dir outputs/scicite_training --batch-size 8 --max-length 256

foreach ($dataset in 'poem','clinc150') {
    & $python scripts/run_type1.py --dataset-dir "data/processed/type1/$dataset" --output-dir "outputs/$dataset"
}
& $python scripts/run_type1.py --dataset-dir data/processed/type1/news --output-dir outputs/news --local-model models/news --training-report outputs/news_training/training_summary.json
& $python scripts/run_type1.py --dataset-dir data/processed/type1_additions/scicite --output-dir outputs/scicite --local-model models/scicite --training-report outputs/scicite_training/training_summary.json --max-length 256
& $python scripts/run_type1.py --dataset-dir data/processed/type1_additions/environmental_claims --output-dir outputs/environmental_claims

.\scripts\download_use_large_v3.ps1
foreach ($dataset in 'poem','clinc150','news','scicite','environmental_claims') {
    & $python scripts/analyze_type1.py --run-dir "outputs/$dataset"
}
```

## Sources

- [TextFooler paper and authors' release](https://github.com/jind11/TextFooler)
- [TextAttack paper](https://aclanthology.org/2020.emnlp-demos.16/) and [TextAttack source](https://github.com/QData/TextAttack)
- Upstream notices for the released TextFooler preprocessing and TextAttack adaptations are in [licenses/](licenses/).
