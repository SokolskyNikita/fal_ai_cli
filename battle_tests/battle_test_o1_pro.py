import os
import unittest
import asyncio

# Import functions directly from FILE.py
from flux_generator import (
    check_api_key,
    get_prompt_hash,
    build_base_args,
    add_ultra_args,
    add_pro_args,
    validate_pro_args,
    validate_model_args,
    setup_output_dir,
    MIN_GUIDANCE,
    MAX_GUIDANCE,
    MIN_STEPS,
    MAX_STEPS,
    MIN_DIM,
    MAX_DIM
)

class TestFILEHelpers(unittest.TestCase):
    def setUp(self):
        # Preserve any existing FAL_KEY and restore after tests
        self.original_fal_key = os.environ.get('FAL_KEY')

    def tearDown(self):
        if self.original_fal_key is None and 'FAL_KEY' in os.environ:
            del os.environ['FAL_KEY']
        elif self.original_fal_key is not None:
            os.environ['FAL_KEY'] = self.original_fal_key

    def test_check_api_key_set(self):
        os.environ['FAL_KEY'] = 'my_test_key'
        key = check_api_key()
        self.assertEqual(key, 'my_test_key')

    def test_check_api_key_not_set(self):
        if 'FAL_KEY' in os.environ:
            del os.environ['FAL_KEY']
        with self.assertRaises(SystemExit):
            _ = check_api_key()

    def test_get_prompt_hash(self):
        prompt = "Test Prompt"
        result = get_prompt_hash(prompt)
        self.assertTrue(len(result) == 8)
        self.assertNotEqual(result, get_prompt_hash("Different"))

    def test_build_base_args(self):
        args = build_base_args(
            prompt="Test prompt",
            out_fmt="png",
            safety="3",
            safety_check=False,
            sync=True
        )
        self.assertEqual(args["prompt"], "Test prompt")
        self.assertEqual(args["output_format"], "png")
        self.assertEqual(args["safety_tolerance"], "3")
        self.assertFalse(args["enable_safety_checker"])
        self.assertTrue(args["sync_mode"])
        self.assertEqual(args["num_images"], 1)

    def test_add_ultra_args(self):
        base = {"prompt": "Ultra test", "num_images": 1}
        updated = add_ultra_args(base, ratio="9:16", raw=True)
        self.assertIn("aspect_ratio", updated)
        self.assertTrue(updated["raw"])
        self.assertEqual(updated["aspect_ratio"], "9:16")

    def test_add_pro_args_with_size(self):
        base = {"prompt": "Pro test", "num_images": 1}
        updated = add_pro_args(base, size="square_hd", w=None, h=None, guidance=5.5, steps=30)
        self.assertIn("image_size", updated)
        self.assertEqual(updated["image_size"], "square_hd")
        self.assertEqual(updated["guidance_scale"], 5.5)
        self.assertEqual(updated["num_inference_steps"], 30)

    def test_add_pro_args_with_dimensions(self):
        base = {"prompt": "Pro dimensions", "num_images": 1}
        updated = add_pro_args(base, size=None, w=300, h=400, guidance=None, steps=None)
        self.assertIn("image_size", updated)
        self.assertDictEqual(updated["image_size"], {"width": 300, "height": 400})

    def test_validate_pro_args_valid(self):
        class Args:
            g = MIN_GUIDANCE
            i = MAX_STEPS
            w = MIN_DIM
            height = MIN_DIM
            size = None
        validate_pro_args(Args())

    def test_validate_pro_args_invalid_guidance(self):
        class Args:
            g = MAX_GUIDANCE + 1
            i = None
            w = None
            height = None
            size = None
        with self.assertRaises(ValueError):
            validate_pro_args(Args())

    def test_validate_pro_args_mismatched_dims(self):
        class Args:
            g = None
            i = None
            w = 300
            height = None
            size = None
        with self.assertRaises(ValueError):
            validate_pro_args(Args())

    def test_validate_model_args_ultra_warnings(self):
        class Args:
            model = "ultra"
            size = "square"
            g = None
            i = None
            w = None
            height = None
            a = None
            raw = None
        # We ensure no exception raised, but a warning might print
        validate_model_args(Args())

    def test_validate_model_args_pro_warnings(self):
        class Args:
            model = "pro"
            a = "16:9"
            raw = True
            size = None
            g = 5
            i = 10
            w = 256
            height = 256
        # We ensure no exception raised, but a warning might print
        validate_model_args(Args())

    def test_setup_output_dir_custom(self):
        custom_dir = "test_output_dir"
        if not os.path.exists(custom_dir):
            os.mkdir(custom_dir)
        path = setup_output_dir("unused_prompt", custom_dir)
        self.assertTrue(os.path.isdir(path))

class TestFILEAsync(unittest.IsolatedAsyncioTestCase):
    # Simple asynchronous checks for structure (does not mock fal_client)
    async def test_run_request_structure(self):
        # run_request calls fal_client.run_async internally,
        # so here we only confirm it awaits and returns a dict or triggers an exception.
        # Without mocking, we won't call external services. We do a partial test only.
        # This minimal test ensures the coroutine can be awaited without syntax errors.
        # We do not provide valid model arguments to avoid external calls.
        with self.assertRaises(Exception):
            from flux_generator import run_request
            await run_request("invalid_model", {"test": "value"})

if __name__ == "__main__":
    unittest.main()
