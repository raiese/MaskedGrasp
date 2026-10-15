# Recovered simulation material

This runner was recovered from the related local ThinkGrasp working tree referenced by the original project and graphical-abstract audit. It is not authenticated as the complete implementation that produced the manuscript. Its current source snapshot is preserved by hashes and documented portability changes.

## Scenes and task instructions

`experiments/manifests/scenes.json` lists each specification, literal target instruction, target indices, asset names, object count, source hash and candidate table association. A scene file has the instruction on line 1, target object indices on line 2, then URDF path and position/orientation values. Indices are mapped to simulation body IDs by the environment; they are not FreeGraspData IDs.

- `testing_cases/`: 10 ordinary-clutter scenes, with 14–15 objects. Tasks include round objects, food, containers, cups, fruit, drinks, thera_med and pear. Candidates for Tables II and IV.
- `heavy_unseen/`: 6 scenes, each with 30 objects. Pairs 00/01 request a ball, 02/03 request a container, 04/05 request fruit. Candidates for Tables II, III and IV.
- Within each heavy pair, the literal instruction is identical. The `_hard` filename and changed assets/poses do not establish the paper's CI condition. A verified contextual-prompt manifest is missing.

Scene specifications are included as small text configurations. The URDFs, meshes, textures, checkpoint files, RGB-D frames and W&B runs are omitted. The custom unseen-object collection has unresolved provenance; upstream assets alone may not suffice.

## Local setup and command template

The recovered environment used Python 3.8.20, torch 1.13.1+cu117, torchvision 0.14.1+cu117 and PyBullet 3.0.7. See `requirements/simulation.txt` and the observed version inventory. Do not combine this with the perception/Qwen profiles without testing.

After obtaining authorized external dependencies from [ThinkGrasp](https://github.com/H-Freax/ThinkGrasp), install its `models/graspnet` source under `simulation/models/graspnet`, build its native operators in the simulation environment, and obtain its checkpoint at `simulation/models/graspnet/logs/log_rs/checkpoint.tar`. Install GroundingDINO and segment-anything separately. Install assets under `simulation/assets/`, retaining per-asset notices. These steps were not executed during this audit.

Use `qwen_service.py` from the repository root in its separate Qwen environment (port 8001), or the recovered `simulation/qwen_remote_server.py` service (default 8008, 4-bit bitsandbytes path, unverified). The default command below explicitly uses port 8001 to match the root service:

```bash
cd simulation
export WANDB_MODE=disabled
export FREEGRASP_GROUNDINGDINO_DIR=../weights/groundingdino
python simulation_main.py \
  --vlm_backend qwen_remote --vlm_api_url http://127.0.0.1:8001/generate \
  --vlm_prompt_variant strict \
  --testing_case_dir testing_cases --num_episode 15 --max_episode_step 50 \
  --sam_ckpt_path ../weights/groundingdino/sam_vit_h_4b8939.pth \
  --groundingdino_config_path langsam/GroundingDINO_SwinB_local_bert.cfg.py \
  --groundingdino_ckpt_path ../weights/groundingdino/groundingdino_swinb_cogcoor.pth
```

For heavy clutter, replace `--testing_case_dir` with `heavy_unseen`. This is a template for the recovered runner, **not** a validated manuscript reproduction command. The explicit 50-step budget reflects current code, and conflicts with the manuscript's stated 15-attempt budget. Do not choose a budget solely to reproduce a reported value.

The runner defaults to headless execution; `--gui` enables visualization. W&B is disabled by default in this candidate. Current scripts use working-directory-relative asset paths, so invoke from `simulation/`.

## Evidence and missing integration

The run manifest links console log directories to sanitized launch arguments where available. It does not infer method labels from dates, directory names, or success rates. Diagnostic records have episode numbers that restart for each case; `(run, episode)` is not a globally unique episode key. Case boundaries need a verified scene order and reset history.

The current runner contains a 3×3 preference path, but no explicit `StateMemory` integration or verified no-memory/no-grid ablation switches. The default local grasp backend differs from the remote FGC geometry backend. The remote branch may consume target identity/instance information supplied by the simulator; such runs need separate oracle-assistance disclosure.

Do not fabricate missing module wiring or rerun commands to claim historical reproducibility. Supply the experiment-specific source commit, method/ablation configuration, scene list, seed, prompts, assets and model revisions first. Recovered source is useful for continuation and debugging but is not a frozen experiment record.
