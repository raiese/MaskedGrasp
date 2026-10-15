# External resources (not bundled)

Consult [LICENSE_REVIEW.md](LICENSE_REVIEW.md) before use. No download has been initiated by packaging these instructions. Pin revisions and record SHA-256 for future runs; current filenames alone cannot establish which version produced historical results.

| Resource | Official download route | Expected local destination |
|---|---|---|
| FreeGraspData | [HF files](https://huggingface.co/datasets/FBK-TeV/FreeGraspData/tree/main/data): two Parquet shards and npz_file.zip | `data/train-00000-of-00002.parquet`, second shard, `data/npz_file/*.npz` |
| MetaGraspNetV2 | [Author download page](https://github.com/maximiliangilles/MetaGraspNet#download-dataset) | User-managed external data directory; no automatic substitution for simulation assets |
| Molmo | [allenai/Molmo-7B-D-0924](https://huggingface.co/allenai/Molmo-7B-D-0924) | `weights/Molmo-7B-D-0924/`; full snapshot including custom model/processor code |
| Qwen | [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | `weights/Qwen2.5-VL-7B-Instruct/` |
| GroundingDINO | [Official repository](https://github.com/IDEA-Research/GroundingDINO), [author checkpoint repository](https://huggingface.co/ShilongLiu/GroundingDINO) | `weights/groundingdino/groundingdino_swinb_cogcoor.pth` and `GroundingDINO_SwinB.cfg.py` |
| SAM ViT-H | [Official SAM repository/model links](https://github.com/facebookresearch/segment-anything#model-checkpoints) | `weights/groundingdino/sam_vit_h_4b8939.pth` |
| Optional BERT text encoder | [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) | HF cache or `weights/groundingdino/bert-base-uncased/` |
| FGC checkpoint | [Official FreeGrasp README, complete pipeline](https://github.com/tev-fbk/FreeGrasp_code#running-on-complete-pipeline-real-world); compare [FGC authors' checkpoint](https://github.com/luyh20/FGC-GraspNet#model) | `logs/checkpoint_fgc.tar`; identity/license of locally used checkpoint unconfirmed |
| ThinkGrasp simulation assets and GraspNet checkpoint | [Official ThinkGrasp setup](https://github.com/H-Freax/ThinkGrasp) | `simulation/assets/`, `simulation/models/graspnet/logs/log_rs/checkpoint.tar` |

Download with the model/dataset card's browser download instructions or its supported Hub client. Do not copy arbitrary unrelated local caches. The `unseen_objects_40` collection needs an exact author-provided source and per-asset permission; the generic upstream ThinkGrasp asset download is not claimed to contain all custom assets.

GroundingDINO/SAM adapters may download missing weights through official clients when invoked. Molmo is configured to load only a complete local snapshot. Use a clean output folder for future inference; static model failures must not overwrite archived experiment records.
