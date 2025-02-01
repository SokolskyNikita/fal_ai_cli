import os
import sys
import json
import hashlib
import argparse
import asyncio
import aiohttp
import aiofiles
import tempfile

import pytest

from flux_generator import (
    check_api_key,
    get_prompt_hash,
    build_base_args,
    add_ultra_args,
    add_pro_args,
    setup_output_dir,
    validate_pro_args,
    validate_model_args,
    run_request,
    submit_request,
    process_single_image,
    download_image,
    download_all_images,
    generate_images,
    cleanup,
    setup_argument_parser,
)

# Synchronous tests

def test_check_api_key_exists(monkeypatch):
    monkeypatch.setenv("FAL_KEY", "dummy_key")
    assert check_api_key() == "dummy_key"

def test_check_api_key_missing(monkeypatch):
    monkeypatch.delenv("FAL_KEY", raising=False)
    with pytest.raises(SystemExit):
        check_api_key()

def test_get_prompt_hash():
    test_prompt = "test"
    expected = hashlib.sha256(test_prompt.encode('utf-8')).hexdigest()[:8]
    assert get_prompt_hash(test_prompt) == expected

def test_build_base_args():
    args = build_base_args("prompt", out_fmt="jpeg", safety="3", safety_check=True, sync=True)
    assert args["prompt"] == "prompt"
    assert args["num_images"] == 1
    assert args["output_format"] == "jpeg"
    assert args["safety_tolerance"] == "3"
    assert args["enable_safety_checker"] is True
    assert args["sync_mode"] is True

def test_add_ultra_args():
    base = {}
    res = add_ultra_args(base, "16:9", True)
    assert res.get("aspect_ratio") == "16:9"
    assert res.get("raw") is True

def test_add_pro_args_with_dimensions():
    base = {}
    res = add_pro_args(base, None, 512, 512, 5.0, 25)
    assert isinstance(res.get("image_size"), dict)
    assert res["image_size"]["width"] == 512
    assert res["image_size"]["height"] == 512
    assert res["guidance_scale"] == 5.0
    assert res["num_inference_steps"] == 25

def test_add_pro_args_with_size():
    base = {}
    res = add_pro_args(base, "square_hd", None, None, None, None)
    assert res.get("image_size") == "square_hd"

def test_setup_output_dir(tmp_path, monkeypatch):
    # Redirect os.getcwd() to the temporary directory
    monkeypatch.setattr(os, "getcwd", lambda: str(tmp_path))
    prompt = "test prompt"
    out_dir = setup_output_dir(prompt)
    assert os.path.exists(out_dir)
    # The metadata file is created later; check it doesn't exist immediately.
    meta = os.path.join(out_dir, "metadata.json")
    assert not os.path.exists(meta)

def test_validate_pro_args_valid():
    ns = argparse.Namespace(g=10, i=30, w=512, height=512)
    # Should not raise any exception
    validate_pro_args(ns)

def test_validate_pro_args_invalid_guidance():
    ns = argparse.Namespace(g=0.5, i=30, w=512, height=512)
    with pytest.raises(ValueError):
        validate_pro_args(ns)

def test_validate_pro_args_invalid_steps():
    ns = argparse.Namespace(g=10, i=60, w=512, height=512)
    with pytest.raises(ValueError):
        validate_pro_args(ns)

def test_validate_pro_args_missing_dimensions():
    ns = argparse.Namespace(g=10, i=30, w=512, height=None)
    with pytest.raises(ValueError):
        validate_pro_args(ns)

def test_validate_pro_args_invalid_dimensions():
    ns = argparse.Namespace(g=10, i=30, w=100, height=100)
    with pytest.raises(ValueError):
        validate_pro_args(ns)

def test_validate_model_args_pro(monkeypatch, capsys):
    ns = argparse.Namespace(model="pro", g=10, i=30, w=512, height=512, a=None, raw=False)
    validate_model_args(ns)
    ns_ultra = argparse.Namespace(model="ultra", g=10, i=30, w=512, height=512, a="16:9", raw=True)
    validate_model_args(ns_ultra)
    captured = capsys.readouterr().out
    assert "Warning" in captured

def test_setup_argument_parser():
    parser = setup_argument_parser()
    args = parser.parse_args(["-p", "test", "-n", "2", "--model", "pro", "-w", "512", "--height", "512"])
    assert args.prompt == "test"
    assert args.n == 2
    assert args.model == "pro"
    assert args.w == 512
    assert args.height == 512

# Asynchronous tests

@pytest.mark.asyncio
async def test_run_request(monkeypatch):
    async def fake_run_async(model, arguments):
        return {"result": "ok"}
    monkeypatch.setattr("FILE.fal_client.run_async", fake_run_async)
    result = await run_request("pro", {"prompt": "test"})
    assert result == {"result": "ok"}

@pytest.mark.asyncio
async def test_submit_request(monkeypatch):
    class FakeHandler:
        request_id = "1234"
    async def fake_submit_async(endpoint, arguments):
        return FakeHandler()
    async def fake_status_async(endpoint, request_id):
        return {"status": "COMPLETED"}
    async def fake_result_async(endpoint, request_id):
        return {"result": "completed"}
    monkeypatch.setattr("FILE.fal_client.submit_async", fake_submit_async)
    monkeypatch.setattr("FILE.fal_client.status_async", fake_status_async)
    monkeypatch.setattr("FILE.fal_client.result_async", fake_result_async)
    result = await submit_request("endpoint", {"prompt": "test"}, 0)
    assert result == {"result": "completed"}

@pytest.mark.asyncio
async def test_process_single_image_subscribe(monkeypatch):
    class FakeInProgress:
        logs = [{"message": "log1"}]
    async def fake_subscribe_async(endpoint, arguments):
        return FakeInProgress()
    monkeypatch.setattr("FILE.fal_client.subscribe_async", fake_subscribe_async)
    result = await process_single_image("endpoint", {"prompt": "test"}, "subscribe")
    assert hasattr(result, "logs")
    assert result.logs[0]["message"] == "log1"

@pytest.mark.asyncio
async def test_process_single_image_run(monkeypatch):
    async def fake_run_async(model, arguments):
        return {"result": "run"}
    monkeypatch.setattr("FILE.fal_client.run_async", fake_run_async)
    result = await process_single_image("endpoint", {"prompt": "test"}, "run")
    assert result == {"result": "run"}

@pytest.mark.asyncio
async def test_download_image(monkeypatch, tmp_path):
    # Create a fake aiohttp response object
    class FakeResponse:
        def __init__(self, status, content):
            self.status = status
            self._content = content
        async def read(self):
            return self._content
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            pass
    class FakeSession:
        async def get(self, url):
            return FakeResponse(200, b"image data")
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            pass
    fname = tmp_path / "test.jpg"
    fake_session = FakeSession()
    result = await download_image(fake_session, "http://example.com/image.jpg", str(fname))
    assert result == str(fname)
    with open(fname, "rb") as f:
        data = f.read()
    assert data == b"image data"

@pytest.mark.asyncio
async def test_download_all_images(monkeypatch, tmp_path):
    async def fake_download_image(session, url, fname):
        async with session.get(url) as resp:
            if resp.status == 200:
                async with aiofiles.open(fname, 'wb') as f:
                    await f.write(b"data")
                return fname
    monkeypatch.setattr("FILE.download_image", fake_download_image)
    urls_and_files = [
        ("http://example.com/img1.jpg", str(tmp_path / "img1.jpg")),
        ("http://example.com/img2.jpg", str(tmp_path / "img2.jpg"))
    ]
    results = await download_all_images(urls_and_files)
    assert all(os.path.exists(fname) for fname in results)

@pytest.mark.asyncio
async def test_generate_images(monkeypatch, tmp_path):
    async def fake_process_single_image(endpoint, arguments, mode):
        return {"images": [{"url": "http://example.com/12345678.jpg"}]}
    async def fake_save_generated_images(result, out_dir):
        return os.path.join(out_dir, "12345678.jpg")
    monkeypatch.setattr("FILE.process_single_image", fake_process_single_image)
    monkeypatch.setattr("FILE.save_generated_images", fake_save_generated_images)
    out_dir = str(tmp_path / "output")
    os.makedirs(out_dir, exist_ok=True)
    results = await generate_images(
        prompt="test",
        count=2,
        out_dir=out_dir,
        model="pro",
        size="square_hd",
        w=512,
        h=512,
        guidance=10,
        steps=30,
        out_fmt="jpeg",
        safety="3",
        safety_check=True,
        seed=100,
        sync=True,
        mode="run"
    )
    assert len(results) == 2

@pytest.mark.asyncio
async def test_cleanup(monkeypatch, tmp_path):
    out_dir = tmp_path / "cleanup_test"
    out_dir.mkdir()
    non_empty = out_dir / "non_empty.txt"
    empty_file = out_dir / "empty.txt"
    non_empty.write_text("data")
    empty_file.write_text("")
    await cleanup([{"dummy": "result"}], str(out_dir))
    meta_file = out_dir / "metadata.json"
    assert meta_file.exists()
    assert not empty_file.exists()
