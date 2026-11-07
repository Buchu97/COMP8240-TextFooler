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

### Constructed sentiment pilots

The two Type 2 pools comprise sentences collected through [TMDB](https://developer.themoviedb.org/docs/faq) and 500 synthetic movie-review sentences generated individually by a language model. The TMDB collector uses a locally configured `TMDB_API_READ_ACCESS_TOKEN` or `TMDB_API_KEY`. A fresh collection can differ as the source changes. This project uses the TMDB API but is not endorsed or certified by TMDB.

```powershell
& $python scripts/collect_tmdb_reviews.py --output private/tmdb_collection --target 500
```

The following commands use locally held, authorised review exports for that collection. The frozen TMDB pilot required two matching human labels. The frozen generated pilot used one completed human review per eligible sentence. Both used a pinned Rotten Tomatoes sentiment checkpoint. The release contains aggregate results and the import pipeline. Exact pilot replay requires the corresponding original candidate files and completed human review exports.

```powershell
& $python -c "from huggingface_hub import snapshot_download; snapshot_download('textattack/bert-base-uncased-rotten-tomatoes', revision='2aeb2b03a543f2eb1f203ca40676fbaf27a3d4fd', cache_dir='data/cache/textattack', allow_patterns=['config.json','pytorch_model.bin','special_tokens_map.json','tokenizer_config.json','vocab.txt'])"
& $python -c "from textattack.shared import WordEmbedding; from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder; WordEmbedding.counterfitted_GLOVE_embedding(); UniversalSentenceEncoder().encode(['A short movie review.'])"
& $python scripts/import_type2_reviews.py --samples private/tmdb_collection/samples.csv --metadata private/tmdb_collection/sample_metadata.csv --reviews private/tmdb_reviewer_1.csv private/tmdb_reviewer_2.csv --output-dir outputs/tmdb_import --min-reviewers 2
& $python scripts/prepare_type2_evaluation.py --import-dir outputs/tmdb_import --output-dir data/processed/tmdb_pilot --experiment-tag TYPE2_PILOT_TMDB --min-reviewers 2 --allow-partial
& $python scripts/run_type2.py --dataset-dir data/processed/tmdb_pilot --output-dir outputs/tmdb_reviewed_pilot --allow-partial --dry-run
```

The asset commands use the cache environment variables set by `setup.ps1` in the current shell. Remove `--dry-run` to run the attack after verification. To evaluate a generated-review pilot, use its authorised candidate and reviewer exports and set `--min-reviewers 1` at both import and preparation. The semantic-similarity analysis uses the same `analyze_type1.py` command.

## Semantic evaluation

The judge compares each successful attack with its original text. One request evaluates the pair for meaning, label preservation and naturalness. Two independent requests classify the texts using the task's labels. The judge sees neither the target classifier's outputs nor the source label. Naturalness is reported separately from meaning-and-label acceptance.

Prepare the five Type 1 runs and export an H100 job bundle. The evaluation reserves five development pairs and selects up to 50 final successful pairs per dataset. The protocol also generates controls.

```powershell
& $python scripts/llm_judge.py prepare --sources notes/judge_sources_type1.json --output-dir outputs/type1_judge --final-per-dataset 50 --audit-size 50
& $python scripts/llm_judge.py plan --prepared outputs/type1_judge --phase final
& $python scripts/llm_judge.py export-h100 --prepared outputs/type1_judge --output-dir outputs/type1_h100 --model Qwen/Qwen3.8-27B --model-revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 --include-final
```

On the H100, open PowerShell in the exported bundle directory. Setup creates its own virtual environment and model cache. The frozen Type 1 run used the following request counts.

```powershell
py -3.11 .\h100_judge.py setup
.\.venv\Scripts\python.exe .\h100_judge.py run --phase development --max-jobs 75
.\.venv\Scripts\python.exe .\h100_judge.py run --phase control --max-jobs 24
.\.venv\Scripts\python.exe .\h100_judge.py run --phase final --max-jobs 750
```

Copy the returned `results` directory into the original bundle and import the responses. For schema failures, `scripts/judge_repair.py` prepares an auditable repair and import. It retains raw responses and supports format-only retries with the same pinned model.

```powershell
$bundle = 'outputs/type1_h100'
$phaseFolders = @{development='h100_development'; control='h100_controls'; final='h100_final'}
foreach ($phase in 'development','control','final') {
    $bundleFile = Join-Path $bundle "$($phaseFolders[$phase])/bundle.json"
    & $python scripts/llm_judge.py import-local --prepared outputs/type1_judge --bundle $bundleFile --results-dir "$bundle/results/$phase" --evaluation outputs/type1_judgements
}
& $python scripts/llm_judge.py summarize --prepared outputs/type1_judge --evaluation outputs/type1_judgements --report-dir outputs/type1_judge_report --phase final
```

The Type 2 pilots use the same protocol with [their own source configuration](notes/judge_sources_type2.json). The data, request bundles, raw responses and review keys are held locally.

## Results

The tables in [results/](results/) describe the verified 3 November 2026 snapshot.

- [Original-data comparison](results/original_replication.csv) covers all 18 main comparisons. Clean accuracy matched the paper to one decimal place in 14 settings. The BERT/Fake News baseline remained 59.5% versus 97.8% in the paper. The legacy and modern loaders agreed on all 1,000 released Fake News inputs.
- [Sentence-similarity ablations](results/sentence_ablation.csv) cover the two paired BERT experiments. Their cosine comparison uses inputs that both variants attacked successfully.
- [Type 1 attack results](results/type1_attacks.csv) cover the five additional datasets. The [class-level breakdown](results/type1_per_class.csv) covers SciCite and Environmental Claims.
- [Type 1 judge results](results/type1_judge.csv) count 88 accepted, 120 rejected and 42 uncertain out of 250 sampled prediction changes. These are judge decisions on a balanced sample of successful attacks. The [judge checks](results/type1_judge_checks.json) record control outcomes and response validation.
- [Type 2 attack results](results/type2_pilots.csv) cover the 212 agreed TMDB examples and the 235 reviewed generated examples in the partial pilots. [TMDB agreement](results/type2_review_agreement.csv) is measured on 250 overlapping reviews before adjudication. Type 2 semantic results are pending in this snapshot.

Clean and after-attack accuracy use all examples. Attack success rate divides successful attacks by the number of initially correct examples. Mean changed words and post-hoc USE-Large cosine are calculated over successful attacks. USE-Large cosine is an automatic similarity measure. The paper-reported similarity in the original-data comparison uses a separate procedure.

## Sources

- [TextFooler paper and authors' release](https://github.com/jind11/TextFooler)
- [TextAttack paper](https://aclanthology.org/2020.emnlp-demos.16/) and [TextAttack source](https://github.com/QData/TextAttack)
- Upstream notices for the released TextFooler preprocessing and TextAttack adaptations are in [licenses/](licenses/).
