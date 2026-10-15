import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from evaluate_target_ids import call_qwen_remote

class ServiceTransportTest(unittest.TestCase):
    def test_image_is_sent_as_bytes_without_shared_path(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'image.png'
            Image.new('RGB',(2,2),(1,2,3)).save(path)
            def fake_urlopen(request,timeout):
                payload=json.loads(request.data)
                self.assertTrue(payload['image_path'].startswith('data:image/png;base64,'))
                decoded=base64.b64decode(payload['image_path'].split(',',1)[1])
                self.assertEqual(decoded,path.read_bytes())
                return io.BytesIO(b'{"text":"3"}')
            with patch('urllib.request.urlopen',side_effect=fake_urlopen):
                self.assertEqual(call_qwen_remote('http://localhost/generate','target',str(path)),'3')

if __name__=='__main__':unittest.main()
