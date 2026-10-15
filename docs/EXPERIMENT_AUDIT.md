# Experiment and manuscript audit

Audit date: 2026-10-15. The source is the local manuscript draft and the two existing working trees, including uncommitted changes. No experiment was rerun to improve agreement with the manuscript. Table entries below are transcriptions or arithmetic from saved records, not newly measured model performance.

## Table I: counts and meaning of ±

`FreeGrasp_code/data/output1/out_*_split{0,1,2}.json` contains 18 files, each with 50 records. All 900 result keys match the corresponding local Parquet subset, split, scene, target and instruction; no missing, extra or duplicate keys were found. This verifies coverage and arithmetic, not the accuracy of the stored success flags or the identity of the model that produced them. There is no immutable run configuration/model revision tied to these files.

The six means round to the manuscript's **FreeGrasp** row. This is a numerical association, not proof that a faithful baseline was run. The current working tree uses a Qwen service, while upstream FreeGrasp evaluated GPT-4o. The 6 unsuffixed `out_*.json` files are older, separate outputs and were not pooled with the 18 split-specific files.

| Subset | Success counts / 50, splits 0/1/2 | Recomputed mean | Population SD (ddof=0) | Sample SD (ddof=1) | Manuscript FreeGrasp |
|---|---|---:|---:|---:|---|
| Easy, unambiguous | 32 / 39 / 35 | 0.706667 | 0.057349 | 0.070238 | 0.71 ± 0.73 |
| Easy, ambiguous | 32 / 32 / 35 | 0.660000 | 0.028284 | 0.034641 | 0.66 ± 0.83 |
| Medium, unambiguous | 8 / 7 / 4 | 0.126667 | 0.033993 | 0.041633 | 0.13 ± 0.40 |
| Medium, ambiguous | 6 / 5 / 5 | 0.106667 | 0.009428 | 0.011547 | 0.11 ± 0.94 |
| Hard, unambiguous | 2 / 3 / 3 | 0.053333 | 0.009428 | 0.011547 | 0.05 ± 0.94 |
| Hard, ambiguous | 6 / 3 / 2 | 0.073333 | 0.033993 | 0.041633 | 0.07 ± 0.40 |

For split accuracies a₀, a₁, a₂, the mean is (a₀+a₁+a₂)/3; population SD is sqrt(sum((aᵢ−mean)²)/3), and sample SD uses denominator 2. These are SDs across instruction variants, not confidence intervals, standard errors, or independent random-seed trials. The original `calculate_SR.py` uses the population definition and prints both quantities as percentages.

All quantities must use the same units. For example, the first row can be reported as **0.706667 ± 0.057349** or **70.6667% ± 5.7349 percentage points** using population SD. It cannot be justified as 0.71 ± 0.73. The printed 0.94 entries match population SD in percentage points for two subsets, but the other ± entries do not match either SD convention. A simple universal percent/proportion conversion does not fix the table.

The newer `data/output/out_easy_no_ambi_split0.json` has only **23/50 expected records, with 12 successes**. Its observed fraction is 12/23 = 0.521739; it must not replace a full split, and no three-split mean is reported for this collection. The original statistics script points to this incomplete directory rather than `output1`.

## The other saved “SSR” file is a different metric

`ThinkGrasp/freegrasp_ssr_results.json` has 900 records. The corresponding `evaluate_freegrasp_ssr.py` computes `predicted_gt_id in ground_truth_ids`, with no predicted mask or IoU. The prompt asks for the described target and discourages selecting blockers; this can also conflict with FreeGraspData's first-object-to-remove labels in occluded scenes. Treat these as target-ID membership results, not segmentation SSR or physical scene completion.

| Subset | Success counts / 50 | Mean | Population SD | Sample SD | Manuscript MaskedGrasp |
|---|---|---:|---:|---:|---|
| Easy, unambiguous | 49 / 49 / 48 | 0.973333 | 0.009428 | 0.011547 | 0.97 ± 1.15 |
| Easy, ambiguous | 39 / 42 / 35 | 0.773333 | 0.057349 | 0.070238 | 0.77 ± 0.04 |
| Medium, unambiguous | 8 / 6 / 9 | 0.153333 | 0.024944 | 0.030551 | 0.20 ± 0.06 |
| Medium, ambiguous | 6 / 4 / 5 | 0.100000 | 0.016330 | 0.020000 | 0.15 ± 0.02 |
| Hard, unambiguous | 3 / 3 / 3 | 0.060000 | 0.000000 | 0.000000 | 0.13 ± 0.00 |
| Hard, ambiguous | 3 / 6 / 3 | 0.080000 | 0.028284 | 0.034641 | 0.09 ± 0.83 |

Only the first two means align after rounding. The first ±1.15 resembles **sample SD expressed in percentage points**; this is a clue, not an established provenance. The other entries cannot be reconstructed from these records. A complete MaskedGrasp segmentation-SSR record set has not been identified.

## Baseline provenance

The [official FreeGrasp paper, arXiv v1, Table I and Section V-A](https://arxiv.org/html/2503.13082v1) reports instruction-variant mean/SD and calls SSR **Segmentation Success Rate**, with IoU ≥ 0.5. Its ThinkGrasp means are 0.63, 0.46, 0.13, **0.16**, 0.05, 0.15. The manuscript matches five entries but reports **0.10** for Medium/ambiguous. This resemblance does not establish whether values were quoted or independently rerun.

That official version's FreeGrasp SSR means are 0.64, 0.64, 0.40, 0.35, 0.13, 0.13, which differ from this manuscript. The manuscript claims all baselines were reimplemented under identical settings, but the inspected records do not establish this claim. Supply each baseline's exact implementation commit, checkpoint/model version, prompt, split definition, random seeds, full predictions and aggregation script. Confirm whether another publication version or rerun is intended; do not silently substitute cited values.

The local Parquet has **900 instruction records, 300 unique scene/target pairs, and 291 distinct scene IDs**. Describing it as 900 independent scenes overstates the physical scene count. Three splits represent alternate instructions for the target scenarios.

## Tables II–IV: evidence mapping

| Manuscript item | Located material | Verification status |
|---|---|---|
| II, ordinary clutter | `simulation/testing_cases/`: 10 scene specifications; candidate run summaries in `simulation_recomputed.json` | No run-to-VLG/OVGrasp/MaskedGrasp mapping or full matching metric triple found |
| II, heavy clutter | `simulation/heavy_unseen/`: 6 scene specifications with 30 objects each | Assets withheld; no authenticated method/configuration mapping |
| III, six heavy tasks | Scene pairs 00/01 (ball), 02/03 (container), 04/05 (fruit) | Similar task families; CI correspondence unverified |
| IV, no 3×3 / no memory / full | Memory utility and grid preference logic exist in separate paths | No archived ablation configuration or matching result set identified |

Across the related simulation project, 33 run directories and 19 step-diagnostic files contain **11,672 action records**. Nine runs have at least one `results/caseN.txt`. There are 27 run directories linked to launch metadata through an explicit log-directory reference in the console output. These links record Qwen/remote configurations, not manuscript method identities. Sanitized launch arguments and checksums are exported; W&B account metadata is excluded.

The full run arithmetic is in `experiments/summaries/simulation_recomputed.json`. Representative complete-looking runs have macro success 0.906667 and 0.913333 over 10 cases, and 0.655556 and 0.677778 over 6 cases. They do not reconstruct the manuscript's 0.933/0.778 full-system results. No numeric resemblance is used to relabel a run as an ablation.

`caseN.txt` stores instruction followed by success rate, average steps, average steps in successful episodes, and reward. The original writer uses **1000 as a sentinel when no episode succeeds**. The new summarizer converts that value to null for success-conditioned statistics and never averages it as a real step count. The macro mean of case-level successful-step means is not the same as pooling all successful episodes; pooled values are left unresolved without authenticated episode counts.

Specific unresolved inconsistencies:

1. The manuscript describes 15 action attempts, but the current simulation runner defaults to 50 steps. Reported heavy-clutter averages (e.g. 23.51 and 32.98) exceed 15. Clarify episode count versus step budget and archive actual launch arguments.
2. Both files in each heavy-scene pair have the **same literal instruction**. The `_hard` suffix alone does not prove the paper's contextual-instruction (CI) condition. Supply contextual prompts and explicit assignment.
3. The runner uses `models.graspnet` by default; the separate remote geometry path uses `models.FGC_graspnet`. The statement that all methods share the identical grasp backbone is not established.
4. The memory class supports history and guidance, but the inspected simulation runner does not call it. The static path calls `update_state_memory(target_name=..., occlusion=None)` without supplying execution outcomes; this does not establish the paper's state-graph updates, topology pruning or feedback loop. No missing mechanism was invented during packaging.
5. `run.py` currently accepts manual target information (and can use a GT instance mask in its internal optional branch); it bypasses automatic language reasoning. The public geometry CLI requires explicit target inputs and exposes no GT-mask option. Document any oracle-assisted historical run separately.
6. Current simulation target-hit logic estimates the grasped object using the highest object's height after a grasp, rather than independently verified object contact. Historical source snapshots and post-lift observations are missing, limiting physical-success verification.
7. Table III's printed six Ours rates average to approximately 0.778, consistent with Table II arithmetically, but there are no identified raw records proving those six numbers. This internal consistency is not experimental verification.
8. No authenticated VLG/OVGrasp run artifacts or no-memory/no-grid run labels were found in the inspected project roots. The manuscript calls the second baseline OVGrasp while reference [40] names OVGNet; confirm intended implementation/name.
9. The manuscript names an Intel RealSense L515 and UR5, while current simulation code selects `RealSenseD435.CONFIG` and `assets/ur5e/ur5e.urdf`. Confirm the actual camera/robot configuration used for the reported runs.
10. The PyBullet scene files reference simplified/unseen object assets. Their mapping to MetaGraspNetV2 object IDs, dataset version and asset licenses has not been established. Do not present these scene files alone as proof of a MetaGraspNetV2 physical benchmark.

## Reproduction of this audit

The code candidate's count CSV contains no benchmark instructions. Recompute both SD definitions using:

```bash
python scripts/summarize_reasoning.py --counts experiments/summaries/reasoning_counts.csv
```

In the **local review bundle only**, the retained records are outside the candidate:

```bash
python scripts/export_reasoning_counts.py \
  --segmentation-dir ../withheld_materials/records/freegrasp/output1 \
  --segmentation-dir ../withheld_materials/records/freegrasp/output \
  --target-id-json ../withheld_materials/records/target_id/freegrasp_ssr_results.json \
  --output outputs/recomputed_counts.csv
python scripts/summarize_simulation.py --results-root ../withheld_materials/records/simulation
```

`source_files.json` records source hashes. Per-record flags have not been rerun against original model outputs/masks, so the results above verify saved-record arithmetic and dataset membership only. Original statistics and results remain untouched.
