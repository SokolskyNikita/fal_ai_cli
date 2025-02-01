import os
import sys
import json
import asyncio
import argparse
import hashlib
import shutil

import pytest
import pytest_asyncio

from flux_generator import (
    check_api_key,
    get_prompt_hash,
    build_base_args,
    add_ultra_args,
    add_pro_args,
    setup_output_dir,
    validate_pro_args,
    validate_model_args,
    setup_argument_parser,
    run_request,
    submit_request,
    process_single_image,
    save_generated_images,
    download_image,
    download_all_images
)

# --- Synchronous Tests ---

def test_get_prompt_hash():
    prompt = "test prompt"
    expected = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
    assert get_prompt_hash(prompt) == expected

def test_build_base_args():
    args = build_base_args("hello", out_fmt="png", safety="3", safety_check=True, sync=False)
    assert args["prompt"] == "hello"
    assert args["num_images"] == 1
    assert args["output_format"] == "png"
    assert args["safety_tolerance"] == "3"
    assert args["enable_safety_checker"] is True
    assert args["sync_mode"] is False

def test_add_ultra_args():
    base = {"prompt": "test"}
    updated = add_ultra_args(base.copy(), ratio="16:9", raw=True)
    assert updated["aspect_ratio"] == "16:9"
    assert updated["raw"] is True

def test_add_pro_args_with_dimensions():
    base = {"prompt": "test"}
    updated = add_pro_args(base.copy(), size=None, w=512, h=512, guidance=5.0, steps=10)
    assert updated["image_size"] == {"width": 512, "height": 512}
    assert updated["guidance_scale"] == 5.0
    assert updated["num_inference_steps"] == 10

def test_add_pro_args_with_size():
    base = {"prompt": "test"}
    updated = add_pro_args(base.copy(), size="square_hd", w=None, h=None, guidance=None, steps=None)
    assert updated["image_size"] == "square_hd"

def test_setup_output_dir(tmp_path, monkeypatch):
    # Force current working directory to tmp_path for isolation.
    monkeypatch.setcwd(tmp_path)
    prompt = "output test"
    out_dir = setup_output_dir(prompt)
    assert os.path.isdir(out_dir)
    # Clean up the created generated directory.
    shutil.rmtree(os.path.join(tmp_path, "generated"))

def test_validate_pro_args_valid():
    ns = argparse.Namespace(w=512, height=512, g=5.0, i=10)
    # Should not raise an error for valid parameters.
    validate_pro_args(ns)

def test_validate_pro_args_invalid_guidance():
    ns = argparse.Namespace(w=512, height=512, g=100, i=10)
    with pytest.raises(ValueError):
        validate_pro_args(ns)

def test_validate_pro_args_incomplete_dimensions():
    ns = argparse.Namespace(w=512, height=None, g=5.0, i=10)
    with pytest.raises(ValueError):
        validate_pro_args(ns)

def test_validate_model_args_pro():
    # For pro model, ultra params (a/raw) are ignored but should not raise an exception.
    ns = argparse.Namespace(model="pro", a="16:9", raw=False, w=512, height=512, g=5.0, i=10)
    validate_model_args(ns)

def test_setup_argument_parser():
    parser = setup_argument_parser()
    args = parser.parse_args(["-p", "test prompt", "-n", "2", "--model", "pro", "-q", "subscribe"])
    assert args.prompt == "test prompt"
    assert args.n == 2
    assert args.model == "pro"
    assert args.q == "subscribe"

def test_check_api_key(monkeypatch):
    # Test when FAL_KEY is set.
    monkeypatch.setenv("FAL_KEY", "dummy_key")
    assert check_api_key() == "dummy_key"
    # Test when FAL_KEY is not set: expect SystemExit.
    monkeypatch.delenv("FAL_KEY", raising=False)
    with pytest.raises(SystemExit):
        check_api_key()

# --- Asynchronous Tests ---

@pytest_asyncio.fixture
def dummy_fal_client(monkeypatch):
    async def dummy_run_async(model, arguments):
        return {"dummy_run": True}

    async def dummy_submit_async(endpoint, arguments):
        class DummyHandler:
            request_id = "1234"
        return DummyHandler()

    async def dummy_status_async(endpoint, request_id):
        return {"status": "COMPLETED"}

    async def dummy_result_async(endpoint, request_id):
        return {"dummy_result": True}

    async def dummy_subscribe_async(endpoint, arguments):
        return {"dummy_subscribe": True}

    monkeypatch.setattr("FILE.fal_client.run_async", dummy_run_async)
    monkeypatch.setattr("FILE.fal_client.submit_async", dummy_submit_async)
    monkeypatch.setattr("FILE.fal_client.status_async", dummy_status_async)
    monkeypatch.setattr("FILE.fal_client.result_async", dummy_result_async)
    monkeypatch.setattr("FILE.fal_client.subscribe_async", dummy_subscribe_async)

@pytest.mark.asyncio
async def test_run_request(dummy_fal_client):
    result = await run_request("pro", {"arg": "value"})
    assert result == {"dummy_run": True}

@pytest.mark.asyncio
async def test_submit_request(dummy_fal_client):
    result = await submit_request("dummy_endpoint", {"arg": "value"}, poll=0)
    assert result == {"dummy_result": True}

@pytest.mark.asyncio
async def test_process_single_image_subscribe(dummy_fal_client):
    result = await process_single_image("dummy_endpoint", {"arg": "value"}, mode="subscribe")
    assert result == {"dummy_subscribe": True}

@pytest.mark.asyncio
async def test_process_single_image_submit(dummy_fal_client):
    result = await process_single_image("dummy_endpoint", {"arg": "value"}, mode="submit")
    assert result == {"dummy_result": True}

@pytest.mark.asyncio
async def test_process_single_image_run(dummy_fal_client):
    result = await process_single_image("dummy_endpoint", {"arg": "value"}, mode="run")
    assert result == {"dummy_run": True}

@pytest_asyncio.fixture
def dummy_download_all(monkeypatch):
    async def dummy_download_all_images(urls_and_files):
        return ["dummy_file.jpg"]
    monkeypatch.setattr("FILE.download_all_images", dummy_download_all_images)

@pytest.mark.asyncio
async def test_save_generated_images_no_images(tmp_path):
    # When result is empty or missing "images", function should return None.
    result = await save_generated_images({}, str(tmp_path))
    assert result is None

@pytest.mark.asyncio
async def test_save_generated_images_with_images(tmp_path, dummy_download_all):
    dummy_url = "http://example.com/abcd1234_image.jpeg"
    result_dict = {"images": [{"url": dummy_url}]}
    out_dir = str(tmp_path / "images")
    result = await save_generated_images(result_dict, out_dir)
    assert result == "dummy_file.jpg"
    assert os.path.isdir(out_dir)

# Dummy classes to simulate aiohttp session and response for download_image.
class DummyResponse:
    def __init__(self, status, data):
        self.status = status
        self._data = data
    async def read(self):
        return self._data
    async def __aenter__(self):
        return self
    async def __aexit__(self, exc_type, exc, tb):
        pass

class DummySession:
    def __init__(self, status, data):
        self.status = status
        self._data = data
    async def get(self, url):
        return DummyResponse(self.status, self._data)
    async def __aenter__(self):
        return self
    async def __aexit__(self, exc_type, exc, tb):
        pass

@pytest.mark.asyncio
async def test_download_image_success(tmp_path):
    dummy_data = b"image data"
    session = DummySession(200, dummy_data)
    fname = str(tmp_path / "test.jpg")
    result = await download_image(session, "http://dummy", fname)
    with open(fname, "rb") as f:
        content = f.read()
    assert content == dummy_data
    assert result == fname

@pytest.mark.asyncio
async def test_download_image_failure(tmp_path):
    session = DummySession(404, b"")
    fname = str(tmp_path / "test.jpg")
    result = await download_image(session, "http://dummy", fname)
    assert result is None
    assert not os.path.exists(fname)

@pytest.mark.asyncio
async def test_download_all_images(monkeypatch):
    async def dummy_download_image(session, url, fname):
        return fname
    monkeypatch.setattr("flux_generator.download_image", dummy_download_image)
    urls_and_files = [("http://dummy", "file1.jpg"), ("http://dummy2", "file2.jpg")]
    results = await download_all_images(urls_and_files)
    assert results == ["file1.jpg", "file2.jpg"]
