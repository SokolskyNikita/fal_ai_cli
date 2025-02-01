import os
import sys
import json
import hashlib
import argparse
import asyncio

import pytest

from flux_generator import (
    check_api_key,
    get_prompt_hash,
    build_base_args,
    add_ultra_args,
    add_pro_args,
    setup_argument_parser,
    setup_output_dir,
    validate_pro_args,
    validate_model_args,
    download_all_images,
    cleanup
)

# --- Synchronous function tests ---

def test_get_prompt_hash():
    prompt = "Test prompt"
    hash_value = get_prompt_hash(prompt)
    expected = hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:8]
    assert hash_value == expected
    assert len(hash_value) == 8

def test_build_base_args():
    prompt = "Sample prompt"
    args = build_base_args(prompt, out_fmt="jpeg", safety="3", safety_check=True, sync=False)
    assert args["prompt"] == prompt
    assert args["num_images"] == 1
    assert args["output_format"] == "jpeg"
    assert args["safety_tolerance"] == "3"
    assert args["enable_safety_checker"] is True
    assert args["sync_mode"] is False

def test_add_ultra_args():
    base_args = {"prompt": "ultra test", "num_images": 1}
    updated = add_ultra_args(base_args.copy(), ratio="16:9", raw=True)
    assert updated.get("aspect_ratio") == "16:9"
    assert updated.get("raw") is True
    # When ratio is None, should not add aspect_ratio.
    updated_none = add_ultra_args(base_args.copy(), ratio=None, raw=False)
    assert "aspect_ratio" not in updated_none
    assert updated_none.get("raw") is False

def test_add_pro_args():
    base_args = {"prompt": "pro test", "num_images": 1}
    updated = add_pro_args(base_args.copy(), size="square_hd", w=None, h=None, guidance=5.0, steps=30)
    # When size is provided and w,h are None.
    assert updated.get("image_size") == "square_hd"
    assert updated.get("guidance_scale") == 5.0
    assert updated.get("num_inference_steps") == 30

    # When width and height are provided.
    updated_wh = add_pro_args(base_args.copy(), size=None, w=512, h=512, guidance=None, steps=None)
    assert updated_wh.get("image_size") == {"width": 512, "height": 512}
    assert "guidance_scale" not in updated_wh
    assert "num_inference_steps" not in updated_wh

def test_setup_argument_parser():
    parser = setup_argument_parser()
    # Check that required mutually exclusive group exists by verifying that either -p or -f must be provided.
    args = parser.parse_args(["-p", "Hello world", "--model", "pro"])
    assert args.prompt == "Hello world"
    assert args.model == "pro"

def test_setup_output_dir(tmp_path):
    prompt = "Temporary prompt"
    custom_dir = tmp_path / "custom_output"
    custom_dir_str = str(custom_dir)
    # When a custom directory is provided.
    out_dir = setup_output_dir(prompt, custom_dir=custom_dir_str)
    assert os.path.isdir(out_dir)
    # When no custom dir is provided, folder is created under generated.
    out_dir2 = setup_output_dir(prompt)
    base_dir = os.path.join(os.getcwd(), "generated")
    assert out_dir2.startswith(base_dir)
    # Cleanup created directories
    if os.path.isdir(out_dir):
        for root, dirs, files in os.walk(out_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(out_dir)
    if os.path.isdir(out_dir2):
        for root, dirs, files in os.walk(out_dir2, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(out_dir2)

def test_validate_pro_args_valid():
    ns = argparse.Namespace(g=10, i=20, w=512, height=512)
    # Should not raise an exception.
    validate_pro_args(ns)

def test_validate_pro_args_invalid_guidance():
    ns = argparse.Namespace(g=100, i=20, w=512, height=512)
    with pytest.raises(ValueError):
        validate_pro_args(ns)

def test_validate_pro_args_missing_dimension():
    ns = argparse.Namespace(g=5, i=10, w=512, height=None)
    with pytest.raises(ValueError):
        validate_pro_args(ns)

def test_validate_model_args_ultra_warning(capfd):
    ns = argparse.Namespace(model="ultra", size="square_hd", g=5, i=10, w=512, height=512, a="16:9", raw=False)
    # For ultra model, pro parameters are ignored; no exception should be raised.
    validate_model_args(ns)
    captured = capfd.readouterr().out
    assert "Warning: Pro model params ignored for Ultra model" in captured

def test_validate_model_args_pro_warning(capfd):
    ns = argparse.Namespace(model="pro", a="16:9", raw=False, g=5, i=10, w=512, height=512)
    # For pro model, ultra parameters are ignored; check for warning message.
    validate_model_args(ns)
    captured = capfd.readouterr().out
    assert "Warning: Ultra model params ignored for Pro model" in captured

def test_check_api_key_set(monkeypatch):
    monkeypatch.setenv('FAL_KEY', 'dummy_key')
    key = check_api_key()
    assert key == 'dummy_key'

def test_check_api_key_not_set(monkeypatch, capsys):
    monkeypatch.delenv('FAL_KEY', raising=False)
    with pytest.raises(SystemExit) as e:
        check_api_key()
    assert e.value.code == 1
    captured = capsys.readouterr().out
    assert "Error: FAL_KEY environment variable not set." in captured

# --- Asynchronous function tests ---

@pytest.mark.asyncio
async def test_download_all_images_empty():
    result = await download_all_images([])
    assert result == []

@pytest.mark.asyncio
async def test_cleanup(tmp_path):
    # Create a temporary directory with one empty file and one non-empty file.
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    empty_file = out_dir / "empty.txt"
    non_empty_file = out_dir / "non_empty.txt"
    empty_file.write_text("")
    non_empty_file.write_text("content")
    # Create a dummy results list.
    results = [{"dummy": "data"}]
    metadata_path = out_dir / "metadata.json"
    await cleanup(results, str(out_dir))
    # Check that metadata file is created and empty file is removed.
    assert metadata_path.exists()
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    assert metadata == results
    assert not empty_file.exists()
    assert non_empty_file.exists()
