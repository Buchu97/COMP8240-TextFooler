# COMP8240 TextFooler replication

This project replicates 15 classification settings and three BERT entailment settings from [Jin et al. (2020)](https://ojs.aaai.org/index.php/AAAI/article/view/6311) and evaluates TextFooler on new text-classification tasks. The primary runs use TextAttack 0.3.10 with up to 50 replacement candidates and a minimum word-embedding cosine similarity of 0.7. The paired ablations remove only the sentence-similarity constraint.

## Quick verification for markers

From a fresh checkout, run this command with Python 3.11 on `PATH`. It reconciles the published aggregates and tests review import, evaluation preparation and judge scoring. Both checks use the Python standard library and the files in this checkout.

```powershell
python scripts/verify_release_results.py --fixtures
```

Attack success rates are 322/390 (82.56%) for TMDb and 351/478 (73.43%) for synthetic reviews. The final judge accepted 31/50 and 21/50 sampled successful attacks, respectively. On the separate 50-pair human audit, the judge and reviewer agreed on meaning preservation in 39 cases. The five-example BERT/MR run below is the smallest replay using the released source data and a real model. Exact Type 2 replay requires authorised review exports and model assets.

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

Recount the BERT/Fake News baseline and the BERT/IMDb ranking probes from completed runs. The optional preprocessing comparison also runs both clean-inference conditions with the same Fake News checkpoint and inputs. The [released attack](https://github.com/jind11/TextFooler/blob/6a56cb10f1055e00a8fbc6882f289e91cfe60f4e/attack_classification.py#L464-L467) calls the generic cleaner by default. The [Fake News training reader](https://github.com/jind11/TextFooler/blob/6a56cb10f1055e00a8fbc6882f289e91cfe60f4e/BERT/run_classifier.py#L404-L429) skips it.

```powershell
& $python scripts/replication_diagnostics.py --runs outputs/replication --authors-data repo/TextFooler-official/data --compare-fake-preprocessing
```

After the full BERT/IMDb run, replay eight evenly spaced eligible rows to count substitutions rejected by POS and sentence-similarity filters. This uses the completed run's snapshots and local CUDA and USE v3 assets. The replay verifies each row against its saved attack.

```powershell
& $python scripts/replay_imdb_query_stages.py --run-dir outputs/replication/original_bert_imdb_full --sample-dir data/processed/released/imdb --output outputs/imdb_query_stage_replay.json
```

### Word-similarity cutoff checks

The paper states a minimum word cosine of 0.7. The [released classification code](https://github.com/jind11/TextFooler/blob/6a56cb10f1055e00a8fbc6882f289e91cfe60f4e/attack_classification.py#L211-L218) uses 0.5. Each paired check uses the same 1,000 inputs, released checkpoint, seed and remaining attack settings. From a disposable checkout, use the prepared AG News and MR splits and checkpoints from the full setup above. Leave word-ranking streaming off at both cutoffs.

```powershell
$samples = [ordered]@{'original-bert-ag' = 'ag'; 'original-lstm-ag' = 'ag'; 'original-bert-mr' = 'mr'}
foreach ($cutoff in '0_7','0_5') {
    if ($cutoff -eq '0_5') {
        git apply notes/bert_ag_word_delta_0_5.patch
        if ($LASTEXITCODE -ne 0) { throw 'Could not apply the 0.5 cutoff patch.' }
    }
    foreach ($model in $samples.Keys) {
        & $python scripts/replication_runs.py run --sample-dir "data/processed/released/$($samples[$model])" --output-dir "outputs/cutoff/${model}_delta_$cutoff" --model $model --seed 8240
        if ($LASTEXITCODE -ne 0) { throw "Cutoff run failed for $model at $cutoff." }
    }
}
```

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

### Constructed sentiment datasets

The Type 2 pools contain 500 source-derived TMDb passages and 500 synthetic movie-review sentences drafted individually using a language model on 6 October 2026. The synthetic pool spans ten linguistic categories and balances 250 intended positive and 250 intended negative labels. The reviewers worked independently in separate tabs and could not see each other's answers. Neither reviewer consulted the metadata tab. A separate local construction file stores the intended labels. Both reviewers' synthetic labels matched all 500 intended labels.

The [synthetic construction record](results/type2_synthetic_construction.json) gives the frozen file hashes, category counts and drafting provenance.

The TMDb collector uses a locally configured `TMDB_API_READ_ACCESS_TOKEN` or `TMDB_API_KEY`. A fresh collection can differ as the source changes. This project uses the TMDB API but is not endorsed or certified by TMDB. [TMDb API FAQ](https://developer.themoviedb.org/docs/faq)

```powershell
& $python scripts/collect_tmdb_reviews.py --output private/tmdb_collection --target 500
```

The 8 November final evaluation required two matching completed human labels per retained sentence in both pools. TMDb uses all 412 agreed passages and excludes 88 disagreements without adjudication. The runner uses its internal `TYPE2_PILOT_` tag and explicit `--allow-partial` switch for this agreed subset. Synthetic reviews use the full 500 agreed rows. Both evaluations use the pinned Rotten Tomatoes sentiment checkpoint. The frozen exports contain no recorded ambiguity flags.

Start with the authorised original candidate, metadata and review exports and run import and preparation in this checkout. Prepared manifests bind to the checkout's execution configuration. Replace the example `private/` paths with paths to the files held locally.

```powershell
& $python -c "from huggingface_hub import snapshot_download; snapshot_download('textattack/bert-base-uncased-rotten-tomatoes', revision='2aeb2b03a543f2eb1f203ca40676fbaf27a3d4fd', cache_dir='data/cache/textattack', allow_patterns=['config.json','pytorch_model.bin','special_tokens_map.json','tokenizer_config.json','vocab.txt'])"
& $python -c "from textattack.shared import WordEmbedding; from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder; WordEmbedding.counterfitted_GLOVE_embedding(); UniversalSentenceEncoder().encode(['A short movie review.'])"
& $python scripts/import_type2_reviews.py --samples private/tmdb_collection/samples.csv --metadata private/tmdb_collection/sample_metadata.csv --reviews private/tmdb_reviewer_1.csv private/tmdb_reviewer_2.csv --output-dir outputs/tmdb_final_import --min-reviewers 2
& $python scripts/prepare_type2_evaluation.py --import-dir outputs/tmdb_final_import --output-dir data/processed/tmdb_agreed --experiment-tag TYPE2_PILOT_TMDB_AGREED_20261108 --min-reviewers 2 --allow-partial
& $python scripts/run_type2.py --dataset-dir data/processed/tmdb_agreed --output-dir outputs/tmdb_final --allow-partial --dry-run

& $python scripts/import_type2_reviews.py --samples private/synthetic_collection/samples.csv --metadata private/synthetic_collection/sample_metadata.csv --reviews private/synthetic_reviewer_1.csv private/synthetic_reviewer_2.csv --output-dir outputs/synthetic_final_import --min-reviewers 2
& $python scripts/prepare_type2_evaluation.py --import-dir outputs/synthetic_final_import --output-dir data/processed/synthetic_final --experiment-tag TYPE2_SYNTHETIC_FINAL_20261108 --min-reviewers 2
& $python scripts/run_type2.py --dataset-dir data/processed/synthetic_final --output-dir outputs/synthetic_final --dry-run
```

The asset commands use the cache environment variables set by `setup.ps1` in the current shell. The dry runs should report 412 TMDb and 500 synthetic rows. Remove `--dry-run` to execute the attacks. Download USE-Large v3 with `scripts/download_use_large_v3.ps1` and run `scripts/analyze_type1.py --run-dir` on each completed attack directory before preparing the judge inputs.

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

### Final Type 2 judge run

Type 2 uses the same protocol. Its source configuration points to the two final attack directories. The judge selected five development and 50 final successful pairs per dataset with seed 8240. Three protocol controls bring the plan to 339 requests at three requests per pair. [Type 2 source configuration](notes/judge_sources_type2.json)

```powershell
& $python scripts/llm_judge.py prepare --sources notes/judge_sources_type2.json --output-dir outputs/type2_judge --seed 8240 --development-per-dataset 5 --final-per-dataset 50 --audit-size 50
& $python scripts/llm_judge.py export-h100 --prepared outputs/type2_judge --output-dir outputs/type2_h100 --model Qwen/Qwen3.8-27B --model-revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 --include-final
```

In the exported bundle directory on the H100, run the development, control and final phases in order.

```powershell
py -3.11 .\h100_judge.py setup
.\.venv\Scripts\python.exe .\h100_judge.py run --phase development --max-jobs 30
.\.venv\Scripts\python.exe .\h100_judge.py run --phase control --max-jobs 9
.\.venv\Scripts\python.exe .\h100_judge.py run --phase final --max-jobs 300
```

Copy the returned `results` directory into the original `outputs/type2_h100` bundle. From the repository root, audit the original responses and prepare any format retries.

```powershell
& $python scripts/judge_repair.py prepare --prepared outputs/type2_judge --pack outputs/type2_h100 --results outputs/type2_h100/results --output-dir outputs/type2_format_audit
```

For the frozen run's 54 format retries, copy `outputs/type2_format_audit/judge_format_repair` into the original H100 bundle. From that bundle's root, run the repair with its pinned model environment.

```powershell
$repairPlan = Get-Content -Raw .\judge_format_repair\repair_plan.json | ConvertFrom-Json
.\.venv\Scripts\python.exe -I .\judge_format_repair\hf_judge_repair.py --job-root . --max-jobs (@($repairPlan.jobs).Count)
```

Copy the resulting `judge_format_repair/results` directory back under `outputs/type2_format_audit/judge_format_repair`. If preparation reports zero requests needing generation, omit the H100 retry step and omit the --retry-results argument below. The import writes phase summaries under `outputs/type2_judge_verified/reports/`. The project holds the data, bundles and raw responses locally.

```powershell
& $python scripts/judge_repair.py import --prepared outputs/type2_judge --pack outputs/type2_h100 --results outputs/type2_h100/results --repair-dir outputs/type2_format_audit --retry-results outputs/type2_format_audit/judge_format_repair/results --output-dir outputs/type2_judge_verified
```

With the completed reviewer CSV and verified local judge responses, the comparison can be recalculated without running Qwen again.

```powershell
& $python scripts/llm_judge.py summarize --prepared outputs/type2_judge --evaluation outputs/type2_judge_verified/evaluations/final --phase final --human-review private/type2_attack_pair_review.csv --report-dir outputs/type2_human_comparison
```

## Results

The tables in [results/](results/) include the final Type 2 evaluation frozen on 8 November 2026 and the two human audits recorded on 9 November 2026.

- [Original-data comparison](results/original_replication.csv) covers all 18 main comparisons. Clean accuracy matched the paper to one decimal place in 14 settings. BERT/Fake News reached 59.5% clean accuracy with the released attack cleaner, versus 97.8% in the paper. Bypassing that cleaner raised clean accuracy to 99.9% on the same rows and checkpoint. The 50.2% after-attack result uses cleaned inputs. The [replication diagnostics](results/replication_diagnostics.json) also count the BERT/IMDb ranking probes under both procedures.
- The [IMDb query-stage replay](results/imdb_query_stage_replay.json) attributes 727 rejected substitutions to POS and sentence-similarity filtering before scoring. The count applies to the eight matched TextAttack trajectories.
- The paired cutoff checks brought after-attack accuracy closer to the paper in all three settings. At 0.5, BERT/AG News reached 11.0% (29.8% at 0.7, 12.5% in the paper), LSTM/AG News reached 3.2% (15.3% at 0.7, 3.8% in the paper), and BERT/MR reached 9.5% (24.8% at 0.7, 11.5% in the paper). Clean predictions matched within each comparison. [BERT/AG News](results/bert_ag_word_threshold.json) · [LSTM/AG News and BERT/MR](results/word_cutoff_followup.json)
- [Sentence-similarity ablations](results/sentence_ablation.csv) cover the two paired BERT experiments. Their cosine comparison uses inputs that both variants attacked successfully.
- [Type 1 attack results](results/type1_attacks.csv) cover the five additional datasets. The [class-level breakdown](results/type1_per_class.csv) covers SciCite and Environmental Claims.
- [Type 1 judge results](results/type1_judge.csv) count 88 accepted, 120 rejected and 42 uncertain out of 250 sampled prediction changes. These are judge decisions on a balanced sample of successful attacks. The [judge checks](results/type1_judge_checks.json) record control outcomes and response validation.
- The 3 November Type 2 pilots covered 212 agreed TMDb and 235 reviewed synthetic examples. The TMDb pilot agreement used 250 overlapping reviews. [Pilot attacks](results/type2_pilots.csv), [pilot agreement](results/type2_review_agreement.csv)
- The final Type 2 evaluations cover 412 agreed TMDb passages and all 500 reviewed synthetic sentences. Both pools have 500 overlapping reviews. Agreement was 82.4% (Cohen's kappa 0.596) for TMDb and 100% (kappa 1.000) for synthetic reviews. [Final attacks](results/type2_final_attacks.csv), [per-class results](results/type2_final_per_class.csv), [annotation checks](results/type2_final_checks.json)
- The final Type 2 judge accepted 31 TMDb and 21 synthetic attacks for meaning and label preservation out of 50 sampled successful attacks per dataset, with zero uncertain decisions. All three controls passed. The 339 usable responses include 285 original responses and 54 format-only retries with the pinned model. [Final judge results](results/type2_final_judge.csv), [judge checks](results/type2_final_checks.json)
- One reviewer and Qwen agreed on meaning preservation for 39 of 50 sampled Type 2 attack pairs (78%, Cohen's kappa 0.56). Reviewer 1 rated 40 of 50 sampled TMDb passages as clear, six as mixed, three as neutral and one as plot-only. The release contains aggregate audit results. Reviewer exports, source identifiers and pair-level judgments are held separately. [Type 2 human audits](results/type2_human_audit.json)
- The full TMDb screen found 293 clear passages. The attack succeeded on 234 of 287 initially correct clear passages (81.53%). Both original reviewers agreed on all 100 synthetic labels in separately shuffled text-only packets. [Data-quality checks](results/type2_data_quality_summary.json)
- For clear TMDb passages, equally weighting positive and negative examples gives a 68.46% attack success rate. Keeping one passage per review gives 68.57%. Omitting one film at a time gives a range of 67.47–69.83%. [TMDb source sensitivity](results/tmdb_source_sensitivity.json)
- Of 42 uncertain Type 1 judge decisions, 41 arose because the single-text judgment of the original disagreed with its source label. The summary also compares the paired task-label judgment with the two separately elicited labels. [Judge protocol checks](results/judge_protocol_checks.json)

Clean and after-attack accuracy use all examples. Attack success rate divides successful attacks by the number of initially correct examples. Mean changed words and post-hoc USE-Large cosine are calculated over successful attacks. USE-Large cosine is an automatic similarity measure. The paper-reported similarity in the original-data comparison uses a separate procedure.

## Sources

- [TextFooler paper and authors' release](https://github.com/jind11/TextFooler)
- [TextAttack paper](https://aclanthology.org/2020.emnlp-demos.16/) and [TextAttack source](https://github.com/QData/TextAttack)
- Upstream notices for the released TextFooler preprocessing and TextAttack adaptations are in [licenses/](licenses/).
