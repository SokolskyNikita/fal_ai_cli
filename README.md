# Fal AI Flux Image Generator

A command-line tool for generating images using the Fal AI Flux API. Only Pro and Ultra models are supported for now.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your Fal AI API key:
```bash
export FAL_KEY=your_api_key_here
```

## Basic Usage

Generate a single image:
```bash
./flux_generator.py -p "A serene mountain landscape at sunset"
```

Generate multiple images:
```bash
./flux_generator.py -p "Abstract art in blue tones" --model ultra -n 5
```

Generate from a file of prompts:
```bash
./flux_generator.py -f prompt.txt -n 3
```

Save to a custom directory:
```bash
./flux_generator.py -p "A serene mountain landscape at sunset" -od /path/to/output
``` 

Default output directory is `./generated/`.

## Pro Model Options

Use a preset image size:
```bash
./flux_generator.py -p "City skyline" --size landscape_16_9
```

Available size presets: `landscape_16_9` (default), `landscape_4_3`, `square_hd`, `portrait_4_3`, `portrait_16_9`

Adjust Prompt Guidance and Inference Steps parameters:
```bash
./flux_generator.py -p "Forest path" -g 7.5 -i 40
```

Custom dimensions:
```bash
./flux_generator.py -p "Ocean waves" -w 1024 --height 768
```

## Ultra Model Options

Set aspect ratio:
```bash
./flux_generator.py -p "Mountain vista" --model ultra -a "16:9"
```

Available aspect ratios: `16:9` (default), `21:9`,  `4:3`, `3:2`, `1:1`, `2:3`, `3:4`, `9:16`, `9:21`.

Raw output:
```bash
./flux_generator.py -p "Abstract art" --model ultra --raw
```

Raw mode is more expensive but makes the image more detailed.

## Other Options

Safety level:
```bash
./flux_generator.py -p "Peaceful garden" -s 3
```

Disable safety checker:
```bash
./flux_generator.py -p "Abstract shapes" --no-safety
```

Set random seed:
```bash
./flux_generator.py -p "Cosmic scene" --seed 42
```

Output format:
```bash
./flux_generator.py -p "Desert landscape" -o png
```
Supports 'jpeg' and 'png'. Default is 'jpeg'.

Queue modes:
```bash
./flux_generator.py -p "Mountain lake" -q submit
./flux_generator.py -p "Forest path" -q run
./flux_generator.py -p "Desert landscape" -q subscribe
```

1. **Subscribe** - Submit and wait for result (default, recommended)
2. **Submit** - Submit and poll for result
3. **Run** - Direct execution (not recommended for long-running tasks)

## Battle Tests

Contains unit tests for the main script written in all current OpenAI models, 1-shot.