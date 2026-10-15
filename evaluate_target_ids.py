import argparse
import base64
import json
import os
import re
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

import pandas as pd


TARGET_ID_SYSTEM_PROMPT = (
    "You identify the target object in a numbered bin-picking image.\n"
    "Your only task is to return the one visible object ID that best matches the text description.\n"
    "Match object category and color first.\n"
    "Do not return a nearby object, support object, lower object, or blocker unless it is itself the described target.\n"
    "Output exactly one line containing only the object ID."
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet-files",
        nargs="+",
        default=[
            "data/train-00000-of-00002.parquet",
            "data/train-00001-of-00002.parquet",
        ],
    )
    parser.add_argument(
        "--npz-dir",
        default="data/npz_file",
    )
    parser.add_argument(
        "--molmo-output-dir",
        default="data/output/molmo_output",
    )
    parser.add_argument(
        "--vlm-api-url",
        default="http://127.0.0.1:8008/generate",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/target_id_results.json",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Optional split filter, e.g. 0, 1, or 2",
    )
    return parser.parse_args()


def load_dataset(parquet_files, split=None):
    df = pd.concat([pd.read_parquet(path) for path in parquet_files], ignore_index=True)
    if split is not None:
        df = df[df["split"].astype(str) == str(split)].copy()
    return df.reset_index(drop=True)


def build_target_id_user_text(task_text, labeled_text):
    lines = labeled_text.strip().splitlines()
    visible_ids = []
    for line in lines[1:]:
        parts = line.split()
        if parts:
            visible_ids.append(parts[0])
    return (
        "Task: identify the target object in the numbered image.\n"
        f"Target instruction: {task_text}\n"
        f"Visible candidate IDs: {', '.join(visible_ids)}\n"
        "Numbered objects:\n"
        f"{labeled_text.strip()}\n"
        "Match the described category and color.\n"
        "Do not choose an object only because it is nearby.\n"
        "Return exactly one object ID and nothing else."
    )


def call_qwen_remote(api_url, instruction, image_path, max_new_tokens=64):
    # Transfer image bytes; the service may use a different working directory.
    image_payload = 'data:image/png;base64,' + base64.b64encode(Path(image_path).read_bytes()).decode('ascii')
    payload = json.dumps(
        {
            "instruction": instruction,
            "image_path": image_payload,
            "system_prompt": TARGET_ID_SYSTEM_PROMPT,
            "max_new_tokens": max_new_tokens,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        api_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        body = json.loads(response.read().decode("utf-8"))
    if "error" in body:
        raise RuntimeError(body["error"])
    return body["text"]


def parse_predicted_id(output):
    match = re.search(r"\b(\d+)\b", output)
    return int(match.group(1)) if match else None


def load_molmo_mapping(scene_dir, scene_id):
    id_path = scene_dir / f"{scene_id}_id.txt"
    if not id_path.exists():
        raise FileNotFoundError(f"Missing Molmo id mapping: {id_path}")

    molmo_to_gt = {}
    with open(id_path, "r", encoding="utf-8") as handle:
        for line in handle.readlines()[1:]:
            parts = line.split()
            if len(parts) >= 4:
                molmo_to_gt[int(parts[0])] = int(parts[3])
    return molmo_to_gt, id_path


def compute_metrics(rows):
    total = len(rows)
    correct = sum(1 for row in rows if row["is_correct"])
    return {
        "total": total,
        "correct": correct,
        "ssr": (correct / total) if total else 0.0,
    }


def summarize_results(results):
    summary = {
        "overall": compute_metrics(results),
        "by_difficulty": {},
        "by_ambiguious": {},
        "by_split": {},
        "by_split_difficulty_ambiguious": {},
    }

    difficulty_groups = defaultdict(list)
    ambiguious_groups = defaultdict(list)
    split_groups = defaultdict(list)
    combo_groups = defaultdict(list)

    for row in results:
        difficulty_groups[row["difficulty"]].append(row)
        ambiguious_groups[str(row["ambiguious"])].append(row)
        split_groups[str(row["split"])].append(row)
        combo_key = f"split={row['split']}|difficulty={row['difficulty']}|ambiguious={row['ambiguious']}"
        combo_groups[combo_key].append(row)

    summary["by_difficulty"] = {
        key: compute_metrics(group) for key, group in sorted(difficulty_groups.items())
    }
    summary["by_ambiguious"] = {
        key: compute_metrics(group) for key, group in sorted(ambiguious_groups.items())
    }
    summary["by_split"] = {
        key: compute_metrics(group) for key, group in sorted(split_groups.items())
    }
    summary["by_split_difficulty_ambiguious"] = {
        key: compute_metrics(group) for key, group in sorted(combo_groups.items())
    }
    return summary


def print_summary(summary):
    overall = summary["overall"]
    print(
        f"Overall SSR: {overall['ssr']:.2%} "
        f"({overall['correct']}/{overall['total']})"
    )

    print("\nBy difficulty:")
    for key, metrics in summary["by_difficulty"].items():
        print(f"  {key}: {metrics['ssr']:.2%} ({metrics['correct']}/{metrics['total']})")

    print("\nBy ambiguious:")
    for key, metrics in summary["by_ambiguious"].items():
        print(f"  {key}: {metrics['ssr']:.2%} ({metrics['correct']}/{metrics['total']})")

    print("\nBy split:")
    for key, metrics in summary["by_split"].items():
        print(f"  {key}: {metrics['ssr']:.2%} ({metrics['correct']}/{metrics['total']})")

    print("\nBy split/difficulty/ambiguious:")
    for key, metrics in summary["by_split_difficulty_ambiguious"].items():
        print(f"  {key}: {metrics['ssr']:.2%} ({metrics['correct']}/{metrics['total']})")


def main():
    args = parse_args()
    df = load_dataset(args.parquet_files, split=args.split)
    if args.limit is not None:
        df = df.head(args.limit).copy()

    results = []
    npz_dir = Path(args.npz_dir)
    molmo_root = Path(args.molmo_output_dir)

    for _, row in df.iterrows():
        scene_id = int(row["sceneId"])
        annotation = str(row["annotation"])
        difficulty = str(row["difficulty"])
        ambiguious = bool(row["ambiguious"])
        split = str(row["split"])
        gt_ids = [int(item.strip()) + 1 for item in str(row["groundTruthObjIds"]).split(",")]

        scene_npz = npz_dir / f"{scene_id}.npz"
        scene_dir = molmo_root / f"scene{scene_id}"
        labeled_image = scene_dir / f"{scene_id}.png"

        result = {
            "scene_id": scene_id,
            "annotation": annotation,
            "difficulty": difficulty,
            "ambiguious": ambiguious,
            "split": split,
            "ground_truth_ids": gt_ids,
            "predicted_molmo_id": None,
            "predicted_gt_id": None,
            "is_correct": False,
            "error": None,
        }

        try:
            if not scene_npz.exists():
                raise FileNotFoundError(f"Missing NPZ: {scene_npz}")
            if not labeled_image.exists():
                raise FileNotFoundError(f"Missing labeled image: {labeled_image}")

            molmo_to_gt, id_path = load_molmo_mapping(scene_dir, scene_id)
            with open(id_path, "r", encoding="utf-8") as handle:
                labeled_text = handle.read()

            user_text = build_target_id_user_text(annotation, labeled_text)
            output = call_qwen_remote(
                args.vlm_api_url,
                instruction=user_text,
                image_path=str(labeled_image),
            )
            predicted_molmo_id = parse_predicted_id(output)
            predicted_gt_id = molmo_to_gt.get(predicted_molmo_id)

            result["predicted_molmo_id"] = predicted_molmo_id
            result["predicted_gt_id"] = predicted_gt_id
            result["raw_output"] = output
            result["is_correct"] = predicted_gt_id in gt_ids
        except (FileNotFoundError, RuntimeError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            result["error"] = str(exc)

        results.append(result)

        status = "OK" if result["is_correct"] else "MISS"
        if result["error"]:
            status = "ERR"
        print(
            f"[{status}] scene={scene_id} split={split} difficulty={difficulty} "
            f"ambiguious={ambiguious} pred_molmo={result['predicted_molmo_id']} "
            f"pred_gt={result['predicted_gt_id']} gt={gt_ids}"
        )

    summary = summarize_results(results)
    payload = {
        "summary": summary,
        "results": results,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print("")
    print_summary(summary)
    print(f"\nSaved detailed results to {output_path}")


if __name__ == "__main__":
    main()
