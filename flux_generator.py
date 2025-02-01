#!/usr/bin/env python3
import fal_client
import asyncio
import aiohttp
from tqdm import tqdm
import os
import json
import hashlib
import argparse
from typing import Optional, List, Dict, Any, Tuple
import sys
import aiofiles

# Model params
ASPECT_RATIOS = ["21:9", "16:9", "4:3", "3:2", "1:1", "2:3", "3:4", "9:16", "9:21"]
IMG_SIZES = ["square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9"]
MIN_GUIDANCE, MAX_GUIDANCE = 1, 20
MIN_STEPS, MAX_STEPS = 1, 50
MIN_DIM, MAX_DIM = 256, 14142

SAFETY_LEVELS = ["1", "2", "3", "4", "5", "6"]
OUT_FORMATS = ["jpeg", "png"]
QUEUE_MODES = ["subscribe", "submit", "run"]
MODELS = ["pro", "ultra"]

ENDPOINTS = {
    "pro": "fal-ai/flux-pro/new",
    "ultra": "fal-ai/flux-pro/v1.1-ultra"
}
DEFAULT_MODEL = "pro"
POLL_INTERVAL = 2

def check_api_key() -> str:
    key = os.environ.get('FAL_KEY')
    if not key:
        print("Error: FAL_KEY environment variable not set.")
        print("Please set your Fal.ai API key:")
        print("export FAL_KEY=your_api_key_here")
        sys.exit(1)
    return key

def get_prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:8]

async def run_request(model: str, args: Dict[str, Any]) -> Dict[str, Any]:
    return await fal_client.run_async(model, arguments=args)

def build_base_args(
    prompt: str,
    out_fmt: Optional[str] = None,
    safety: Optional[str] = None,
    safety_check: Optional[bool] = None,
    sync: Optional[bool] = None
) -> Dict[str, Any]:
    args = {
        "prompt": prompt,
        "num_images": 1,
    }
    
    if out_fmt:
        args["output_format"] = out_fmt
    if safety:
        args["safety_tolerance"] = safety
    if safety_check is not None:
        args["enable_safety_checker"] = safety_check
    if sync is not None:
        args["sync_mode"] = sync
        
    return args

def add_ultra_args(args: Dict[str, Any], ratio: Optional[str], raw: bool) -> Dict[str, Any]:
    if ratio:
        args["aspect_ratio"] = ratio
    if raw:
        args["raw"] = raw
    return args

def add_pro_args(
    args: Dict[str, Any],
    size: Optional[str],
    w: Optional[int],
    h: Optional[int],
    guidance: Optional[float],
    steps: Optional[int]
) -> Dict[str, Any]:
    if w is not None and h is not None:
        args["image_size"] = {"width": w, "height": h}
    elif size:
        args["image_size"] = size
    if guidance is not None:
        args["guidance_scale"] = guidance
    if steps is not None:
        args["num_inference_steps"] = steps
    return args

async def submit_request(endpoint: str, args: Dict[str, Any], poll: int) -> Dict[str, Any]:
    handler = await fal_client.submit_async(endpoint, arguments=args)
    while True:
        status = await fal_client.status_async(endpoint, handler.request_id)
        if status["status"] == "COMPLETED":
            return await fal_client.result_async(endpoint, handler.request_id)
        await asyncio.sleep(poll)

async def process_single_image(
    endpoint: str,
    args: Dict[str, Any],
    mode: str,
    poll: int = POLL_INTERVAL
) -> Dict[str, Any]:
    if mode == "subscribe":
        result = await fal_client.subscribe_async(endpoint, arguments=args)
        if isinstance(result, fal_client.InProgress):
            for log in result.logs:
                print(log["message"])
        return result
    elif mode == "submit":
        return await submit_request(endpoint, args, poll)
    else:
        return await run_request(endpoint, args)

async def save_generated_images(result: Dict[str, Any], out_dir: str) -> Optional[str]:
    if not result or "images" not in result:
        return None

    os.makedirs(out_dir, exist_ok=True)
    
    urls_and_files = []
    for img in result["images"]:
        url = img["url"]
        flux_id = url.split("/")[-1].split("_")[0][:8]
        ext = url.split(".")[-1].lower()
        fname = os.path.join(out_dir, f"{flux_id}.{ext}")
        urls_and_files.append((url, fname))
    
    downloaded = await download_all_images(urls_and_files)
    return downloaded[0] if downloaded else None

async def generate_images(
    prompt: str,
    count: int,
    out_dir: str,
    model: str,
    ratio: Optional[str] = None,
    raw: Optional[bool] = None,
    size: Optional[str] = None,
    w: Optional[int] = None,
    h: Optional[int] = None,
    guidance: Optional[float] = None,
    steps: Optional[int] = None,
    out_fmt: Optional[str] = None,
    safety: Optional[str] = None,
    safety_check: Optional[bool] = None,
    seed: Optional[int] = None,
    sync: Optional[bool] = None,
    mode: str = "subscribe"
) -> List[dict]:
    endpoint = ENDPOINTS[model]
    results = []
    failed = 0
    
    base_args = build_base_args(
        prompt=prompt,
        out_fmt=out_fmt,
        safety=safety,
        safety_check=safety_check,
        sync=sync
    )
    
    if model == "ultra":
        base_args = add_ultra_args(base_args, ratio, raw if raw is not None else False)
    else:
        base_args = add_pro_args(base_args, size, w, h, guidance, steps)

    tasks = []
    for i in range(count):
        task_args = base_args.copy()
        if seed is not None:
            task_args["seed"] = seed + i
        tasks.append(process_single_image(endpoint, task_args, mode))

    for result in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating images"):
        try:
            data = await result
            if out_dir:
                await save_generated_images(data, out_dir)
            results.append(data)
        except Exception as e:
            print(f"Error generating image: {e}")
            failed += 1

    if failed:
        print(f"Failed to generate {failed} images")
    
    return results

async def process_generation(args: argparse.Namespace, prompt: str, out_dir: str) -> None:
    gen_args = {
        "prompt": prompt,
        "count": args.n,
        "out_dir": out_dir,
        "model": args.model,
        "sync": args.sync,
        "mode": args.q
    }
    
    if args.o:
        gen_args["out_fmt"] = args.o
    if args.s:
        gen_args["safety"] = args.s
    if args.seed is not None:
        gen_args["seed"] = args.seed
    
    if args.model == "ultra":
        if args.a:
            gen_args["ratio"] = args.a
        if args.raw:
            gen_args["raw"] = args.raw
    else:
        if args.w and args.height:
            gen_args["w"] = args.w
            gen_args["h"] = args.height
        elif args.size:
            gen_args["size"] = args.size
        if args.g is not None:
            gen_args["guidance"] = args.g
        if args.i is not None:
            gen_args["steps"] = args.i
    
    results = await generate_images(**gen_args)
    await cleanup(results, out_dir)
    print(f"Generation completed. Results saved to '{out_dir}' folder.")

async def cleanup(results: List[dict], out_dir: str) -> None:
    async with aiofiles.open(os.path.join(out_dir, "metadata.json"), "w") as f:
        await f.write(json.dumps(results, indent=4))
    
    for root, _, files in os.walk(out_dir):
        for file in files:
            path = os.path.join(root, file)
            if os.path.getsize(path) == 0:
                os.remove(path)
                print(f"Removed empty file: {path}")

async def download_image(session: aiohttp.ClientSession, url: str, fname: str) -> Optional[str]:
    async with session.get(url) as resp:
        if resp.status == 200:
            async with aiofiles.open(fname, 'wb') as f:
                await f.write(await resp.read())
            return fname
    return None

async def download_all_images(urls_and_files: List[Tuple[str, str]]) -> List[Optional[str]]:
    async with aiohttp.ClientSession() as session:
        tasks = [download_image(session, url, fname) for url, fname in urls_and_files]
        return await asyncio.gather(*tasks)

def setup_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate images using Fal AI Flux API")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-p", dest="prompt", help="Text prompt")
    group.add_argument("-f", dest="file", help="Prompt file")
    
    parser.add_argument("--model", choices=MODELS, default="pro",
                      help="Model: pro/ultra")
    
    parser.add_argument("-n", type=int, default=1,
                      help="Number of images")
    parser.add_argument("-o", choices=OUT_FORMATS,
                      help="Format: jpeg/png")
    parser.add_argument("-s", choices=SAFETY_LEVELS,
                      help="Safety level 1-6")
    parser.add_argument("--no-safety", action="store_true",
                      help="Disable safety check")
    parser.add_argument("--seed", type=int,
                      help="Random seed")
    parser.add_argument("--sync", action="store_true",
                      help="Use sync mode")
    parser.add_argument("-od", help="Output dir")
    
    ultra = parser.add_argument_group('Ultra model params')
    ultra.add_argument("-a", choices=ASPECT_RATIOS,
                       default="16:9",
                       help="Aspect ratio")
    ultra.add_argument("--raw", action="store_true",
                       help="Less processed output")
    
    pro = parser.add_argument_group('Pro model params')
    size_group = pro.add_mutually_exclusive_group()
    size_group.add_argument("--size", choices=IMG_SIZES,
                            default="landscape_16_9",
                        help="Size preset")
    pro.add_argument("-w", type=int,
                        help=f"Width {MIN_DIM}-{MAX_DIM}")
    pro.add_argument("--height", type=int,
                        help=f"Height {MIN_DIM}-{MAX_DIM}")
    pro.add_argument("-g", type=float,
                        help=f"Guidance {MIN_GUIDANCE}-{MAX_GUIDANCE}")
    pro.add_argument("-i", type=int,
                        help=f"Inference steps {MIN_STEPS}-{MAX_STEPS}")
    
    parser.add_argument("-q", choices=QUEUE_MODES, default="subscribe",
                      help="Queue: subscribe/submit/run")
    parser.add_argument("--poll", type=int, default=POLL_INTERVAL,
                      help="Poll interval (sec)")
    
    return parser

async def process_all_prompts(args: argparse.Namespace) -> None:
    if args.prompt:
        prompt = args.prompt
        out_dir = setup_output_dir(prompt)
        await process_generation(args, prompt, out_dir)
        return
        
    # Read prompts from file
    with open(args.file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    if not prompts:
        raise ValueError("No valid prompts found in the file")
        
    print(f"Found {len(prompts)} prompts to process")
    for i, prompt in enumerate(prompts, 1):
        print(f"\nProcessing prompt {i}/{len(prompts)}: {prompt}")
        out_dir = setup_output_dir(prompt, args.od)
        await process_generation(args, prompt, out_dir)
    print("\nAll prompts processed successfully")

async def main():
    try:
        check_api_key()
        
        parser = setup_argument_parser()
        args = parser.parse_args()
        validate_model_args(args)
        
        await process_all_prompts(args)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def validate_pro_args(args: argparse.Namespace) -> None:
    if args.g is not None:
        if not MIN_GUIDANCE <= args.g <= MAX_GUIDANCE:
            raise ValueError(f"guidance must be between {MIN_GUIDANCE} and {MAX_GUIDANCE}")
    
    if args.i is not None:
        if not MIN_STEPS <= args.i <= MAX_STEPS:
            raise ValueError(f"steps must be between {MIN_STEPS} and {MAX_STEPS}")
    
    if bool(args.w) != bool(args.height):
        raise ValueError("Both -w and --height must be provided together")
    
    if args.w and args.height:
        if not (MIN_DIM <= args.w <= MAX_DIM and MIN_DIM <= args.height <= MAX_DIM):
            raise ValueError(f"dimensions must be between {MIN_DIM} and {MAX_DIM}")

def validate_model_args(args: argparse.Namespace) -> None:
    if args.model == "ultra":
        if any([args.size, args.g, args.i, args.w, args.height]):
            print("Warning: Pro model params ignored for Ultra model")
    else:
        if args.a or args.raw:
            print("Warning: Ultra model params ignored for Pro model")
        validate_pro_args(args)

def setup_output_dir(prompt: str, custom_dir: Optional[str] = None) -> str:
    if custom_dir:
        os.makedirs(custom_dir, exist_ok=True)
        return custom_dir
        
    base = os.path.join(os.getcwd(), "generated")
    os.makedirs(base, exist_ok=True)
    
    folder = get_prompt_hash(prompt)
    out_dir = os.path.join(base, folder)
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Saving results to folder: {out_dir}")
    return out_dir

if __name__ == "__main__":
    asyncio.run(main())