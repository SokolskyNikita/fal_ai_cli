import unittest
import os
import hashlib
from flux_generator import (
    check_api_key,
    get_prompt_hash,
    build_base_args,
    add_ultra_args,
    add_pro_args
)

class TestFileModule(unittest.TestCase):

    def setUp(self):
        self.prompt = "Test prompt"
        self.api_key = "test_api_key"

    def test_check_api_key_present(self):
        os.environ['FAL_KEY'] = self.api_key
        self.assertEqual(check_api_key(), self.api_key)

    def test_check_api_key_absent(self):
        if 'FAL_KEY' in os.environ:
            del os.environ['FAL_KEY']
        with self.assertRaises(SystemExit):
            check_api_key()

    def test_get_prompt_hash(self):
        expected_hash = hashlib.sha256(self.prompt.encode('utf-8')).hexdigest()[:8]
        self.assertEqual(get_prompt_hash(self.prompt), expected_hash)

    def test_build_base_args_minimal(self):
        expected_args = {
            "prompt": self.prompt,
            "num_images": 1,
        }
        self.assertEqual(build_base_args(self.prompt), expected_args)

    def test_build_base_args_full(self):
        expected_args = {
            "prompt": self.prompt,
            "num_images": 1,
            "output_format": "png",
            "safety_tolerance": "3",
            "enable_safety_checker": True,
            "sync_mode": False
        }
        self.assertEqual(
            build_base_args(
                self.prompt,
                out_fmt="png",
                safety="3",
                safety_check=True,
                sync=False
            ),
            expected_args
        )

    def test_add_ultra_args(self):
        base_args = {"prompt": self.prompt, "num_images": 1}
        expected_args = {
            "prompt": self.prompt,
            "num_images": 1,
            "aspect_ratio": "16:9",
            "raw": True
        }
        self.assertEqual(add_ultra_args(base_args, "16:9", True), expected_args)

    def test_add_pro_args(self):
        base_args = {"prompt": self.prompt, "num_images": 1}
        expected_args = {
            "prompt": self.prompt,
            "num_images": 1,
            "image_size": {"width": 512, "height": 512},
            "guidance_scale": 7.5,
            "num_inference_steps": 25
        }
        self.assertEqual(
            add_pro_args(
                base_args,
                size=None,
                w=512,
                h=512,
                guidance=7.5,
                steps=25
            ),
            expected_args
        )

if __name__ == '__main__':
    unittest.main()
