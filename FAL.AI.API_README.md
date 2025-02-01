# Fal.ai Python Client Library Documentation

This document covers the key APIs from the Fal.ai Python client library that are used in our image generation script.

## Core Concepts

The library provides three main ways to interact with Fal.ai endpoints:

1. **Subscribe** - Submit and wait for result (recommended)
2. **Submit** - Submit and poll for result
3. **Run** - Direct execution (not recommended for long-running tasks)

## Authentication

The library uses the `FAL_KEY` environment variable for authentication:

```python
import os
api_key = os.environ.get('FAL_KEY')  # Must be set before using the library
```

## Async vs Sync Methods

The library provides both synchronous and asynchronous versions of each API:

```python
# Async versions (recommended)
result = await fal_client.subscribe_async(...)
result = await fal_client.submit_async(...)
result = await fal_client.run_async(...)

# Sync versions
result = fal_client.subscribe(...)
result = fal_client.submit(...)
result = fal_client.run(...)
```

The async versions are recommended for better performance and responsiveness, especially when generating multiple images.

## Parallelization Strategy

The library's `num_images` parameter is limited to 1, but we can achieve parallelization by:

1. Using ThreadPoolExecutor for sync methods:
```python
with ThreadPoolExecutor(max_workers=count) as executor:
    futures = []
    for i in range(count):
        future = executor.submit(fal_client.subscribe, ...)
        futures.append(future)
    results = [f.result() for f in futures]
```

2. Using asyncio.gather for async methods:
```python
tasks = []
for i in range(count):
    task = fal_client.subscribe_async(...)
    tasks.append(task)
results = await asyncio.gather(*tasks)
```

## Key APIs Used

### 1. Subscribe Method

```python
result = fal_client.subscribe(
    endpoint_id,      # e.g., "fal-ai/flux-pro/new"
    arguments={       # API parameters
        "prompt": "your prompt",
        "num_images": 1,
        ...
    },
    with_logs=True,  # Enable log output
    on_queue_update=callback_function  # Progress updates
)
```

The `on_queue_update` callback receives updates about request progress:
```python
def on_queue_update(update):
    if isinstance(update, fal_client.InProgress):
        for log in update.logs:
            print(log["message"])
```

### 2. Submit and Poll Method

```python
# Submit request
handler = await fal_client.submit_async(
    endpoint_id,
    arguments={...}
)
request_id = handler.request_id

# Check status
status = await fal_client.status_async(
    endpoint_id, 
    request_id,
    with_logs=True
)

# Get result when complete
result = await fal_client.result_async(
    endpoint_id,
    request_id
)
```

Status response types:
- `Queued` - Request is waiting (includes queue position)
- `InProgress` - Request is being processed (includes logs)
- `Completed` - Request is done (includes result)

### 3. Direct Run Method

```python
result = await fal_client.run_async(
    endpoint_id,
    arguments={...}
)
```

⚠️ Not recommended for production use as it blocks until completion and doesn't handle connection issues.

## Response Format

All methods return results in the same format:

```python
{
    "prompt": str,           # Original prompt
    "images": [{            # List of generated images
        "url": str,         # URL to download image
        "width": int,       # Image width
        "height": int,      # Image height
        "content_type": str # e.g., "image/jpeg"
    }],
    "timings": dict,        # Performance metrics
    "seed": int,           # Used seed value
    "has_nsfw_concepts": [bool]  # Safety check results
}
```

## Error Handling

The library raises `FalClientError` for API-related errors. Common scenarios:
- Invalid API key
- Invalid parameters
- Network issues
- Server errors

## Queue Management

Queue status codes:
- `IN_QUEUE` - Waiting to be processed
- `IN_PROGRESS` - Currently being processed
- `COMPLETED` - Processing finished

## Model-Specific Parameters

### Pro Model
```python
{
    "prompt": str,
    "image_size": str | dict,  # Preset or {"width": int, "height": int}
    "guidance_scale": float,   # Range: 1-20
    "num_inference_steps": int # Range: 1-50
}
```

### Ultra Model
```python
{
    "prompt": str,
    "aspect_ratio": str,  # e.g., "16:9"
    "raw": bool          # Less processed output
}
```

### Common Parameters
```python
{
    "output_format": "jpeg" | "png",
    "safety_tolerance": "1" | "2" | "3" | "4" | "5" | "6",
    "enable_safety_checker": bool,
    "seed": int,
    "sync_mode": bool
}
```

## Image Size Options

### Pro Model Presets
Each preset has specific dimensions optimized for different use cases:

```python
VALID_IMAGE_SIZES = {
    "square_hd": "High-definition square format",
    "square": "Standard square format",
    "portrait_4_3": "Portrait orientation (4:3)",
    "portrait_16_9": "Portrait orientation (16:9)",
    "landscape_4_3": "Landscape orientation (4:3)",
    "landscape_16_9": "Landscape orientation (16:9)"
}
```

### Custom Dimensions
For Pro model, you can specify custom dimensions:
```python
"image_size": {
    "width": int,   # Range: 1-14142
    "height": int   # Range: 1-14142
}
```

### Ultra Model Aspect Ratios
Available aspect ratios with their typical use cases:
```python
VALID_ASPECT_RATIOS = {
    "21:9": "Ultrawide format",
    "16:9": "Standard widescreen",
    "4:3": "Traditional display",
    "3:2": "Classic photography",
    "1:1": "Square format",
    "2:3": "Portrait photography",
    "3:4": "Portrait display",
    "9:16": "Mobile display",
    "9:21": "Mobile ultrawide"
}
```

## Performance Considerations

1. **Queue Modes**:
   - `subscribe`: Best for most cases, handles everything automatically
   - `submit`: Good for long-running jobs with webhook support
   - `run`: Only for quick operations, not recommended for production

2. **Batch Processing**:
   - Use ThreadPoolExecutor for parallel processing
   - Keep track of seeds for reproducibility
   - Handle rate limits and errors gracefully

3. **Memory Management**:
   - Download images in chunks
   - Clean up temporary files
   - Use async downloads for better performance

## Common Patterns

### Reproducible Generation
```python
# Same seed + same prompt = same image
api_args = {
    "prompt": "your prompt",
    "seed": 42
}
```

### Progress Tracking
```python
with tqdm(total=count) as pbar:
    for result in results:
        # Process result
        pbar.update(1)
```

### Error Recovery
```python
try:
    result = await fal_client.subscribe_async(...)
except FalClientError as e:
    if "rate limit" in str(e):
        # Handle rate limiting
    elif "invalid key" in str(e):
        # Handle authentication error
    else:
        # Handle other errors
```

## References

- [Official Python Client Documentation](https://docs.fal.ai/clients/python)
- [API Reference](https://fal-ai.github.io/fal/client/fal_client.html) 