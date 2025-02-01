import os
import sys
import json
import hashlib
import argparse
import tempfile
import asyncio

import pytest

from flux_generator import (
    check_api_key,
    get_prompt_hash,
    build_base_args,
    add_ultra_args,
    add_pro_args,
    setup_argument_parser,
    validate_pro_args,
    validate_model_args,
    setup_output_dir,
    run_request,
    submit_request,
    process_single_image,
    save_generated_images,
    generate_images,
    cleanup
)
import fal_client

# ----- Synchronous Function Tests -----

def test_check_api_key_set(monkeypatch):
    monkeypatch.setenv("FAL_KEY", "testkey")
    key = check_api_key()
    assert key == "testkey"

def test_check_api_key_not_set(monkeypatch):
    monkeypatch.delenv("FAL_KEY", raising=False)
    with pytest.raises(SystemExit):
        check_api_key()

def test_get_prompt_hash():
    prompt = "Test prompt"
    expected = hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:8]
    assert get_prompt_hash(prompt) == expected

def test_build_base_args():
    args = build_base_args("hello", out_fmt="png", safety="3", safety_check=True, sync=False)
    assert args["prompt"] == "hello"
    assert args["num_images"] == 1
    assert args["output_format"] == "png"
    assert args["safety_tolerance"] == "3"
    assert args["enable_safety_checker"] is True
    assert args["sync_mode"] is False

def test_build_base_args_minimal():
    args = build_base_args("hello")
    assert args == {"prompt": "hello", "num_images": 1}

def test_add_ultra_args():
    base = {"prompt": "hello"}
    updated = add_ultra_args(base, "16:9", True)
    assert updated["aspect_ratio"] == "16:9"
    assert updated["raw"] is True

def test_add_pro_args_with_dimensions():
    base = {"prompt": "hello"}
    updated = add_pro_args(base, size=None, w=512, h=512, guidance=5.0, steps=10)
    assert "image_size" in updated
    assert updated["image_size"] == {"width": 512, "height": 512}
    assert updated["guidance_scale"] == 5.0
    assert updated["num_inference_steps"] == 10

def test_add_pro_args_with_size():
    base = {"prompt": "hello"}
    updated = add_pro_args(base, size="square_hd", w=None, h=None, guidance=None, steps=None)
    assert updated["image_size"] == "square_hd"
    assert "guidance_scale" not in updated
    assert "num_inference_steps" not in updated

def test_setup_argument_parser():
    parser = setup_argument_parser()
    args = parser.parse_args(["-p", "test prompt", "--model", "pro", "-n", "3", "-q", "run"])
    assert args.prompt == "test prompt"
    assert args.model == "pro"
    assert args.n == 3
    assert args.q == "run"

def test_validate_pro_args_invalid_guidance():
    ns = argparse.Namespace(g=0.5, i=10, w=512, height=512)
    with pytest.raises(ValueError):
        validate_pro_args(ns)

def test_validate_pro_args_invalid_steps():
    ns = argparse.Namespace(g=5, i=0, w=512, height=512)
    with pytest.raises(ValueError):
        validate_pro_args(ns)

def test_validate_pro_args_mismatched_dimensions():
    ns = argparse.Namespace(g=5, i=10, w=512, height=None)
    with pytest.raises(ValueError):
        validate_pro_args(ns)

def test_validate_pro_args_dimensions_out_of_range():
    ns = argparse.Namespace(g=5, i=10, w=100, height=100)
    with pytest.raises(ValueError):
        validate_pro_args(ns)

def test_validate_pro_args_valid():
    ns = argparse.Namespace(g=5, i=10, w=512, height=512)
    # Should not raise error
    validate_pro_args(ns)

def test_validate_model_args_ultra_with_pro(monkeypatch, capsys):
    ns = argparse.Namespace(model="ultra", size="square_hd", g=5, i=10, w=512, height=512, a=None, raw=False)
    validate_model_args(ns)
    captured = capsys.readouterr().out
    assert "Warning" in captured

def test_validate_model_args_pro_with_ultra(monkeypatch, capsys):
    ns = argparse.Namespace(model="pro", a="16:9", raw=True, g=5, i=10, w=512, height=512, size=None)
    validate_model_args(ns)
    captured = capsys.readouterr().out
    assert "Warning" in captured

def test_setup_output_dir_custom(tmp_path):
    custom_dir = tmp_path / "custom_output"
    prompt = "test prompt"
    out_dir = setup_output_dir(prompt, custom_dir=str(custom_dir))
    assert os.path.exists(out_dir)
    assert out_dir == str(custom_dir)

def test_setup_output_dir_default(monkeypatch, tmp_path):
    monkeypatch.setattr(os, "getcwd", lambda: str(tmp_path))
    prompt = "test prompt"
    out_dir = setup_output_dir(prompt)
    expected_base = os.path.join(str(tmp_path), "generated")
    assert os.path.exists(expected_base)
    assert out_dir.startswith(expected_base)

# ----- Asynchronous Function Tests -----

@pytest.mark.asyncio
async def test_run_request(monkeypatch):
    async def dummy_run_async(model, arguments):
        return {"model": model, "args": arguments}
    monkeypatch.setattr(fal_client, "run_async", dummy_run_async)
    result = await run_request("pro", {"key": "value"})
    assert result["model"] == "pro"
    assert result["args"] == {"key": "value"}

class DummyHandler:
    request_id = "dummy_id"

@pytest.mark.asyncio
async def test_submit_request(monkeypatch):
    async def dummy_submit_async(endpoint, arguments):
        return DummyHandler()
    async def dummy_status_async(endpoint, request_id):
        return {"status": "COMPLETED"}
    async def dummy_result_async(endpoint, request_id):
        return {"result": "done"}
    monkeypatch.setattr(fal_client, "submit_async", dummy_submit_async)
    monkeypatch.setattr(fal_client, "status_async", dummy_status_async)
    monkeypatch.setattr(fal_client, "result_async", dummy_result_async)
    monkeypatch.setattr(asyncio, "sleep", lambda x: asyncio.sleep(0))
    result = await submit_request("endpoint", {"arg": 1}, poll=0)
    assert result == {"result": "done"}

class DummyInProgress:
    logs = [{"message": "log1"}, {"message": "log2"}]

@pytest.mark.asyncio
async def test_process_single_image_subscribe(monkeypatch, capsys):
    async def dummy_subscribe_async(endpoint, arguments):
        return DummyInProgress()
    monkeypatch.setattr(fal_client, "subscribe_async", dummy_subscribe_async)
    result = await process_single_image("endpoint", {"arg": 1}, "subscribe")
    captured = capsys.readouterr().out
    assert "log1" in captured
    assert "log2" in captured
    assert result is not None

@pytest.mark.asyncio
async def test_process_single_image_submit(monkeypatch):
    async def dummy_submit_async(endpoint, arguments):
        return DummyHandler()
    async def dummy_status_async(endpoint, request_id):
        return {"status": "COMPLETED"}
    async def dummy_result_async(endpoint, request_id):
        return {"result": "submitted"}
    monkeypatch.setattr(fal_client, "submit_async", dummy_submit_async)
    monkeypatch.setattr(fal_client, "status_async", dummy_status_async)
    monkeypatch.setattr(fal_client, "result_async", dummy_result_async)
    result = await process_single_image("endpoint", {"arg": 1}, "submit", poll=0)
    assert result == {"result": "submitted"}

@pytest.mark.asyncio
async def test_process_single_image_run(monkeypatch):
    async def dummy_run_request(endpoint, arguments):
        return {"result": "run"}
    # Monkey-patch run_request in the FILE module
    import flux_generator
    monkeypatch.setattr(flux_generator, "run_request", dummy_run_request)
    result = await flux_generator.process_single_image("endpoint", {"arg": 1}, "run")
    assert result == {"result": "run"}

@pytest.mark.asyncio
async def test_save_generated_images(monkeypatch, tmp_path):
    out_dir = str(tmp_path / "images")
    result_data = {
        "images": [
            {"url": "http://example.com/12345678_extra.jpeg"}
        ]
    }
    async def dummy_download_all_images(urls_and_files):
        return ["dummy_file.jpeg"]
    monkeypatch.setattr("FILE.download_all_images", dummy_download_all_images)
    res = await save_generated_images(result_data, out_dir)
    assert res == "dummy_file.jpeg"
    assert os.path.exists(out_dir)

@pytest.mark.asyncio
async def test_generate_images(monkeypatch, tmp_path):
    out_dir = str(tmp_path / "images")
    prompt = "test"
    async def dummy_process_single_image(endpoint, args, mode, poll=2):
        return {"result": "image"}
    monkeypatch.setattr("FILE.process_single_image", dummy_process_single_image)
    async def dummy_save_generated_images(result, out_dir):
        return None
    monkeypatch.setattr("FILE.save_generated_images", dummy_save_generated_images)
    results = await generate_images(
        prompt, 3, out_dir, "pro",
        size="square_hd", w=512, h=512, guidance=5.0, steps=10,
        out_fmt="jpeg", safety="3", safety_check=True, seed=100, sync=True, mode="run"
    )
    assert len(results) == 3
    for r in results:
        assert r == {"result": "image"}

@pytest.mark.asyncio
async def test_cleanup(tmp_path):
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    file_nonempty = out_dir / "nonempty.txt"
    file_empty = out_dir / "empty.txt"
    file_nonempty.write_text("data")
    file_empty.write_text("")
    await cleanup([{"dummy": "data"}], str(out_dir))
    metadata = out_dir / "metadata.json"
    assert metadata.exists()
    assert not file_empty.exists()
    assert file_nonempty.exists()
