# MaskedGrasp — local release candidate

Code and experiment-audit material for the MaskedGrasp manuscript being prepared for submission to **Neurocomputing**. The proposed method combines language-guided target selection, state memory, semantic region constraints, segmentation, and grasp geometry in cluttered scenes.

**Status:** this is a code-first, local review candidate, not a verified reproduction of all manuscript tables. The available implementation contains FreeGrasp-derived reasoning and geometry code, a lightweight memory module, and a separately recovered ThinkGrasp-derived PyBullet runner. The complete memory/feedback integration and the mapping of runs to the claimed methods remain unresolved. See [the table audit](docs/EXPERIMENT_AUDIT.md) before using any number as a paper result.

No benchmark images, annotations, meshes, model weights, raw experiment logs, manuscript PDF, Git history, or tracking-account metadata are bundled. Large experimental materials are deferred until after publication, subject to their original licenses. The small count summaries here are derived from existing records and are not new experiments.

## Layout

```text
MaskedGrasp/
  evaluate.py                 Static mask-IoU evaluation CLI
  evaluate_target_ids.py      Recovered target-ID evaluator (not segmentation SSR)
  molmo_eval.py               Object pointing and numbered visual prompts
  reasoning_eval.py          Mask-IoU success and resumable result writing
  run.py, grasp_model.py      Single-frame grasp geometry, explicit target input
  qwen_service.py            Local reasoning service (port 8001)
  remote_server.py           Geometry service (port 8000)
  utils/                     Memory, prompts, client, camera/grasp utilities
  models/langsam/            Local segmentation adapters; external models excluded
  simulation/                Recovered PyBullet runner and 16 scene definitions
  scripts/                   Entry points, count export, statistical verification
  experiments/manifests/     Source hashes, scenes, run inventory
  experiments/summaries/     Small derived count tables and audits
  configs/                   Environment variables and example camera calibration
  requirements/              Separate dependency profiles and observed versions
  tests/                     Memory, statistics, evaluation-bookkeeping tests
  docs/                      Evidence limits, downloads, licensing, change log
  licenses/                  Preserved upstream notices (not a new blanket license)
```

Commands below run from this directory unless a different working directory is shown. Output directories are new, user-selected directories; never point them at an archived experiment.

## Installation

### Offline audit (no GPU or weights)

Python 3.10+ is sufficient for the count/statistics scripts. Dataset-membership verification and the evaluator unit test also need NumPy, pandas, Pillow and PyArrow:

```bash
python3 -m venv .venv-audit
. .venv-audit/bin/activate
python -m pip install -r requirements/audit.txt
python -m unittest discover -s tests -v
python scripts/summarize_reasoning.py \
  --counts experiments/summaries/reasoning_counts.csv \
  --output outputs/reasoning_recomputed.json
```

### Model inference (installation recipe; not end-to-end verified)

Use separate environments for the legacy perception stack and the Qwen service. The original environments have different Python/PyTorch/Transformers versions; merging them into one environment is not supported by the current evidence.

```bash
conda create -n maskedgrasp python=3.10
conda activate maskedgrasp
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements/perception.txt
python -m pip install 'git+https://github.com/IDEA-Research/GroundingDINO.git'
python -m pip install 'git+https://github.com/facebookresearch/segment-anything.git'
python -m pip install 'git+https://github.com/graspnet/graspnetAPI.git'
```

Acquire the external `models/FGC_graspnet` implementation from the [FreeGrasp code repository](https://github.com/tev-fbk/FreeGrasp_code), retaining its notices. The standalone [FGC-GraspNet](https://github.com/luyh20/FGC-GraspNet) repository has a different layout; it is not a drop-in replacement. Its redistribution/license status needs author clarification; it is deliberately excluded here. After installing that dependency locally:

```bash
(cd models/FGC_graspnet/pointnet2 && python setup.py install)
(cd models/FGC_graspnet/knn && python setup.py install)
```

A CUDA toolkit compatible with PyTorch is required for these extensions. Prebuilt `.so` files are not shipped. Record exact upstream commits and wheel/build versions for a new experiment. [Observed versions and verification limits](docs/VALIDATION.md) are distinct from the proposed installation profile.

For the Qwen service, use a separate Python 3.10 environment, install a matching CUDA PyTorch/torchvision pair, then `python -m pip install -r requirements/qwen.txt`. The included service uses `Qwen2_5_VLForConditionalGeneration`; Transformers 4.43.3 in the legacy perception environment cannot provide that class.

## Data and models

Use the official links in [DOWNLOADS.md](docs/DOWNLOADS.md). In particular:

- [FreeGraspData](https://huggingface.co/datasets/FBK-TeV/FreeGraspData): download the two Parquet shards and `data/npz_file.zip`; unpack to the layout below.
- [MetaGraspNetV2](https://github.com/maximiliangilles/MetaGraspNet): obtain data from the authors' download page. Confirm the version used in the manuscript; the upstream project reports a 2025 alignment correction.
- [Molmo-7B-D-0924](https://huggingface.co/allenai/Molmo-7B-D-0924), [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), and [SAM](https://github.com/facebookresearch/segment-anything): download through their official repositories/model cards.
- The FGC checkpoint is linked by the official FreeGrasp README. Its precise provenance and checkpoint redistribution terms still need confirmation; no weight is included.

```text
data/
  train-00000-of-00002.parquet
  train-00001-of-00002.parquet
  npz_file/<sceneId>.npz
weights/
  Molmo-7B-D-0924/           # Complete model snapshot including processor files
  Qwen2.5-VL-7B-Instruct/
  groundingdino/            # SwinB config/checkpoint, SAM ViT-H, optional local BERT
logs/checkpoint_fgc.tar     # Acquired separately
```

Review the dataset/model terms before downloading. Dataset licenses do not inherit the code license. Set paths and service addresses using `configs/env.example`; legacy `FREEGRASP_*` variable names are retained for compatibility. The camera JSON is an example, not a verified calibration for another camera.

## Running static reasoning and segmentation

In the separate Qwen environment, start the service (loads weights):

```bash
python qwen_service.py --host 127.0.0.1 --port 8001 --model-path weights/Qwen2.5-VL-7B-Instruct
```

In the perception environment:

```bash
export MOLMO_MODEL_PATH=weights/Molmo-7B-D-0924
export FREEGRASP_GROUNDINGDINO_DIR=weights/groundingdino
export FREEGRASP_VILA_BASE_URL=http://127.0.0.1:8001
python evaluate.py --stage localize --data-root data --output-root outputs/new_run
python evaluate.py --stage reasoning --data-root data --output-root outputs/new_run
python scripts/summarize_reasoning.py --raw-dir outputs/new_run
```

Use `--splits 0 --subsets easy_no_ambi` for a smaller reasoning check. `--stage all` runs both stages. Molmo localization prepares all scene IDs in the selected splits; `--subsets` controls the reasoning stage only. `--molmo-output` can select existing numbered images.

Success in this evaluator means a predicted object mask has IoU ≥ 0.5 with at least one annotated ground-truth object mask. It is **segmentation success**, not physical scene completion. The three splits correspond to instruction variants; they are not three random-seed trials. Missing samples/inference errors are recorded as failures in new runs, and errors are saved separately. This bookkeeping differs from the historical evaluator; see [CHANGES.md](docs/CHANGES.md).

The separate target-ID evaluator is preserved for provenance:

```bash
python evaluate_target_ids.py --parquet-files data/train-00000-of-00002.parquet data/train-00001-of-00002.parquet \
  --npz-dir data/npz_file --molmo-output-dir outputs/new_run/molmo_output \
  --vlm-api-url http://127.0.0.1:8001/generate --output-json outputs/target_id_results.json
```

Its historical JSON calls the metric `ssr`, but it checks predicted GT-ID membership and performs no mask-IoU evaluation. Do not combine it with segmentation results.

## Single-frame grasp geometry

This entry point requires an explicitly selected target and writes a pose to the input example directory. It does not claim to run the paper's complete language/memory policy.

```bash
python scripts/infer_grasp.py --input-dir data/my_example \
  --camera-json configs/camera.example.json \
  --target-id 7 --target-class scissor --target-xy 402 542 \
  --checkpoint logs/checkpoint_fgc.tar
```

The example directory must contain `image.png` and `depth.npz` with a `depth` array. Replace all example target/camera values with actual observations. Geometry uses the external FGC implementation and native CUDA operators. Real-robot control scripts and private camera integrations are excluded.

## Simulation and historical records

The recovered runner lives in `simulation/`. Read [SIMULATION.md](docs/SIMULATION.md) for dependencies, assets, commands, scene-to-table candidates and limits. It is a historical ThinkGrasp-derived runner; it has no verified complete MaskedGrasp memory/ablation integration. Do not label a new run as a reproduction of Tables II–IV without resolving that gap.

To recompute simulation summaries after obtaining the retained records:

```bash
python scripts/summarize_simulation.py --results-root /path/to/records/simulation --output outputs/simulation.json
python scripts/verify_dataset.py --data-root data --result-dir /path/to/recorded/output1
```

The public candidate carries per-split count summaries, source hashes and run manifests. Detailed JSON/JSONL records remain in the separate local review area and are deferred. A hash identifies a source artifact; it does not prove the model/configuration that produced it.

## License, attribution and citation

License selection is pending author confirmation. Existing FreeGrasp-derived code remains subject to its [CC BY-NC 4.0 terms](licenses/FreeGrasp-CC-BY-NC-4.0.txt); ThinkGrasp-derived portions retain [MIT notices](licenses/ThinkGrasp-MIT.txt). External GraspNet/FGC components have separate restrictions or unresolved provenance. A blanket MIT/Apache license would not remove those conditions. See [LICENSE_REVIEW.md](docs/LICENSE_REVIEW.md).

This candidate builds on FreeGrasp, ThinkGrasp, FGC-GraspNet, GraspNet, GroundingDINO, SAM, Molmo and Qwen. Please cite their original publications when used. MaskedGrasp author list, final title, publication year, DOI and repository URL must be supplied before adding a final citation; none is invented here.
