import json
import tempfile
import unittest
from pathlib import Path
from scripts.summarize_reasoning import count_records, summarize

class StatisticsTest(unittest.TestCase):
    def test_ddof_and_units(self):
        counts = [dict(collection='test', metric='SSR', subset='easy_no_ambi',
                       split=str(i), total=50, correct=k) for i,k in enumerate([32,39,35])]
        row = summarize(counts)[0]
        self.assertAlmostEqual(row['mean'], 106/150)
        self.assertAlmostEqual(row['population_sd'], 0.05734883511361752)
        self.assertAlmostEqual(row['sample_sd'], 0.07023769168568493)
        self.assertIsNone(summarize(counts[:1])[0]['mean'])
        with self.assertRaises(ValueError):
            summarize(counts + counts[:1])

    def test_reject_duplicate_and_invalid_records(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'result.json'
            for rows in [[], [[0,1,'x',1],[0,1,'x',0]], [[0,1,'x',2]]]:
                path.write_text(json.dumps(rows))
                with self.assertRaises(ValueError):
                    count_records(path)
            path.write_text(json.dumps([[0,1,'x',1],[1,1,'x',0]]))
            self.assertEqual(count_records(path), (2,1))

if __name__ == '__main__':
    unittest.main()
