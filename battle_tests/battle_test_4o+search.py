import unittest
import os
import hashlib
from flux_generator import (
    check_api_key,
    get_prompt_hash,
    build_base_args,
    add_ultra_args,
    add_pro_args,
    setup_output_dir
)

class TestFILEFunctions(unittest.TestCase):

    def setUp(self):
        self.prompt = "Test prompt"
        self.out_fmt = "jpeg"
        self.safety = "3"
        self.safety_check = True
        self.sync = False
        self.ratio = "16:9"
        self.raw = True
        self.size = "square_hd"
        self.width = 1024
        self.height = 768
        self.guidance = 7.5
        self.steps = 25
        self.custom_dir = "custom_output"

    def test_check_api_key_present(self):
        os.environ['FAL_KEY'] = 'dummy_key'
        self.assertEqual(check_api_key(), 'dummy_key')

    def test_check_api_key_absent(self):
        if 'FAL_KEY' in os.environ:
            del os.environ['FAL_KEY']
        with self.assertRaises(SystemExit):
            check_api_key()

    def test_get_prompt_hash(self):
        expected_hash = hashlib.sha256(self.prompt.encode('utf-8')).hexdigest()[:8]
        self.assertEqual(get_prompt_hash(self.prompt), expected_hash)

    def test_build_base_args(self):
        expected_args = {
            "prompt": self.prompt,
            "num_images": 1,
            "output_format": self.out_fmt,
            "safety_tolerance": self.safety,
            "enable_safety_checker": self.safety_check,
            "sync_mode": self.sync
        }
        self.assertEqual(
            build_base_args(
                prompt=self.prompt,
                out_fmt=self.out_fmt,
                safety=self.safety,
                safety_check=self.safety_check,
                sync=self.sync
            ),
            expected_args
        )

    def test_add_ultra_args(self):
        base_args = {"prompt": self.prompt, "num_images": 1}
        expected_args = {
            "prompt": self.prompt,
            "num_images": 1,
            "aspect_ratio": self.ratio,
            "raw": self.raw
        }
        self.assertEqual(add_ultra_args(base_args, self.ratio, self.raw), expected_args)

    def test_add_pro_args(self):
        base_args = {"prompt": self.prompt, "num_images": 1}
        expected_args = {
            "prompt": self.prompt,
            "num_images": 1,
            "image_size": {"width": self.width, "height": self.height},
            "guidance_scale": self.guidance,
            "num_inference_steps": self.steps
        }
        self.assertEqual(
            add_pro_args(
                base_args,
                size=None,
                w=self.width,
                h=self.height,
                guidance=self.guidance,
                steps=self.steps
            ),
            expected_args
        )

    def test_setup_output_dir_custom(self):
        out_dir = setup_output_dir(self.prompt, self.custom_dir)
        self.assertTrue(os.path.exists(out_dir))
        self.assertEqual(out_dir, self.custom_dir)

    def tearDown(self):
        if os.path.exists(self.custom_dir):
            os.rmdir(self.custom_dir)

if __name__ == '__main__':
    unittest.main()
