"""Static mask-IoU evaluation adapted from FreeGrasp; see docs/CHANGES.md."""
import io
import json
from pathlib import Path


def compute_iou(mask_pred, mask_gt):
    import numpy as np
    pred, gt = np.asarray(mask_pred, dtype=bool), np.asarray(mask_gt, dtype=bool)
    if pred.shape != gt.shape:
        raise ValueError(f'Mask shape mismatch: {pred.shape} vs {gt.shape}')
    union = np.logical_or(pred, gt).sum()
    return float(np.logical_and(pred, gt).sum()/union) if union else 0.0


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def process_dataset(df, dataset_root, molmo_output_root, output_json_path, predictor=None):
    import numpy as np
    from PIL import Image
    if predictor is None:
        from utils.utils import get_prediction_result
        predictor = get_prediction_result
    output = Path(output_json_path)
    errors_path = output.with_suffix('.errors.json')
    results = json.loads(output.read_text()) if output.exists() else []
    errors = json.loads(errors_path.read_text()) if errors_path.exists() else []
    expected = {(int(r.sceneId), int(r.queryObjId)+1, r.annotation) for r in df.itertuples()}
    processed = set()
    for result in results:
        if len(result) != 4 or result[3] not in (0, 1):
            raise ValueError('Malformed existing result')
        key = tuple(result[:3])
        if key in processed or key not in expected:
            raise ValueError('Existing output contains duplicates or another subset/split')
        processed.add(key)
    for _, row in df.iterrows():
        key = (int(row['sceneId']), int(row['queryObjId'])+1, row['annotation'])
        if key in processed:
            continue
        correct = 0
        try:
            gt_ids = [int(x)+1 for x in str(row['groundTruthObjIds']).split(',')]
            with np.load(Path(dataset_root)/f'{key[0]}.npz', allow_pickle=False) as data:
                gt_map = data['instances_objects']
            if not any(np.any(gt_map == gt) for gt in gt_ids):
                raise ValueError('No annotated ground-truth object exists in instance map')
            image = Image.open(io.BytesIO(row['image']['bytes'])).convert('RGB')
            pred = predictor(image, Path(molmo_output_root)/f'scene{key[0]}', key[0], key[2])
            if isinstance(pred, dict) and 'error' in pred:
                raise RuntimeError(pred['error'])
            correct = int(any(compute_iou(pred, gt_map == gt) >= 0.5 for gt in gt_ids))
        except Exception as exc:
            errors.append(dict(scene_id=key[0], object_id=key[1], error_type=type(exc).__name__,
                               message=str(exc)))
        results.append([*key, correct])
        processed.add(key)
        _save(output, results)
        _save(errors_path, errors)
    total, correct = len(results), sum(r[3] for r in results)
    return dict(total=total, correct=correct, accuracy=correct/total if total else None,
                errors=len(errors), output_json_path=str(output))


if __name__ == '__main__':
    raise SystemExit('Use: python evaluate.py --help')
