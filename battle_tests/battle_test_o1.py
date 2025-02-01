import os
import pytest
import shutil
from pathlib import Path
import pytest_asyncio
from flux_generator import (
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


def test_get_prompt_hash():
    prompt = "Test prompt"
    result = get_prompt_hash(prompt)
    assert isinstance(result, str)
    assert len(result) == 8
    # Simple check that it's consistently derived from the same prompt
    assert result == get_prompt_hash(prompt)


def test_build_base_args():
    args = build_base_args(
        prompt="A sample prompt",
        out_fmt="png",
        safety="3",
        safety_check=True,
        sync=False
    )
    assert args["prompt"] == "A sample prompt"
    assert args["output_format"] == "png"
    assert args["safety_tolerance"] == "3"
    assert args["enable_safety_checker"] is True
    assert args["sync_mode"] is False
    assert args["num_images"] == 1


def test_add_ultra_args():
    base = {"prompt": "test"}
    result = add_ultra_args(base, ratio="4:3", raw=True)
    assert result["aspect_ratio"] == "4:3"
    assert result["raw"] is True


def test_add_pro_args():
    base = {"prompt": "test"}
    result = add_pro_args(
        base,
        size="square",
        w=None,
        h=None,
        guidance=7.5,
        steps=25
    )
    assert result["image_size"] == "square"
    assert result["guidance_scale"] == 7.5
    assert result["num_inference_steps"] == 25


def test_validate_pro_args_valid():
    class Args:
        g = 5.0
        i = 10
        w = 512
        height = 512
    validate_pro_args(Args)


def test_validate_pro_args_invalid_guidance():
    class Args:
        g = -1
        i = 10
        w = 512
        height = 512
    with pytest.raises(ValueError):
        validate_pro_args(Args)


def test_validate_pro_args_invalid_steps():
    class Args:
        g = 5.0
        i = 1000
        w = 512
        height = 512
    with pytest.raises(ValueError):
        validate_pro_args(Args)


def test_validate_pro_args_unmatched_dimensions():
    class Args:
        g = 5.0
        i = 10
        w = 512
        height = None
    with pytest.raises(ValueError):
        validate_pro_args(Args)


def test_validate_pro_args_invalid_dimension_range():
    class Args:
        g = 5.0
        i = 10
        w = 999999
        height = 999999
    with pytest.raises(ValueError):
        validate_pro_args(Args)


def test_validate_model_args_ultra_ignores_pro():
    class Args:
        model = "ultra"
        size = "square"
        g = None
        i = None
        w = None
        height = None
        a = "16:9"
        raw = True
    # Should print a warning but not fail
    validate_model_args(Args)


def test_validate_model_args_pro_ignores_ultra():
    class Args:
        model = "pro"
        size = None
        g = 5.0
        i = 20
        w = 512
        height = 512
        a = "16:9"
        raw = True
    # Should print a warning but not fail
    validate_model_args(Args)


def test_setup_output_dir(tmp_path):
    test_prompt = "Test directory setup"
    # If custom_dir is provided, it should directly create that path
    custom_dir = tmp_path / "custom_output"
    result_dir = setup_output_dir(test_prompt, str(custom_dir))
    assert os.path.exists(result_dir)
    assert os.path.samefile(result_dir, str(custom_dir))

    # If no custom_dir, it should hash the prompt and create generated/<hash>
    new_dir = setup_output_dir(test_prompt)
    assert os.path.exists(new_dir)
    assert "generated" in new_dir
    shutil.rmtree(new_dir, ignore_errors=True)
