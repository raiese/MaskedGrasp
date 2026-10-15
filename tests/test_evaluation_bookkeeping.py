import io
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from reasoning_eval import process_dataset

class EvaluationBookkeepingTest(unittest.TestCase):
    def test_final_failure_is_saved_and_resume_counts_all_records(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            buffer = io.BytesIO()
            Image.new('RGB', (2,2)).save(buffer, format='PNG')
            rows = [dict(sceneId=i, queryObjId=0, annotation='target', groundTruthObjIds='0',
                         image={'bytes':buffer.getvalue()}) for i in range(2)]
            df = pd.DataFrame(rows)
            np.savez(root/'0.npz', instances_objects=np.ones((2,2), dtype=int))
            predictor = lambda *args: np.ones((2,2), dtype=bool)
            output = root/'out.json'
            report = process_dataset(df, root, root, output, predictor=predictor)
            self.assertEqual((report['total'],report['correct'],report['errors']), (2,1,1))
            self.assertEqual(json.loads(output.read_text())[-1][-1], 0)
            resumed = process_dataset(df, root, root, output, predictor=predictor)
            self.assertEqual(resumed['accuracy'], 0.5)
            self.assertEqual(resumed['errors'], 1)

if __name__ == '__main__':
    unittest.main()
