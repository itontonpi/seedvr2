# ComfyUI-SeedVR2_VideoUpscaler

This repository is an unofficial personal test fork of `numz/ComfyUI-SeedVR2_VideoUpscaler`. It is not the upstream project and is not maintained or supported by the upstream authors.

This repository provides ComfyUI integration for SeedVR2, with a standalone CLI for image and video upscaling.

This repository exposes the SeedVR2 pipeline in two forms:

- ComfyUI custom nodes for interactive workflows
- `inference_cli.py` for single-file, batch, streaming, and multi-GPU runs

![SeedVR2](docs/seedvr_logo.png)

## What It Does

- Upscales images and videos with SeedVR2 diffusion models
- Supports RGB and RGBA inputs in the ComfyUI node path
- Supports 3B and 7B DiT variants, plus GGUF and FP8 model options
- Includes VAE tiling, BlockSwap, tensor offloading, model caching, and optional `torch.compile`
- Can auto-download known model files into the SeedVR2 model directory

## Project Layout

- `src/interfaces/` contains the ComfyUI nodes
- `src/core/` contains pipeline orchestration and model configuration
- `src/optimization/` contains memory, compatibility, GGUF, and BlockSwap logic
- `inference_cli.py` is the standalone entrypoint
- `example_workflows/` contains sample ComfyUI workflows and example inputs

## ComfyUI Nodes

The integration is split into four nodes:

1. `SeedVR2 (Down)Load DiT Model`
2. `SeedVR2 (Down)Load VAE Model`
3. `SeedVR2 Torch Compile Settings`
4. `SeedVR2 Video Upscaler`

Typical wiring:

1. Load or select a DiT model
2. Load the shared VAE
3. Optionally provide compile settings
4. Send an image batch into `SeedVR2 Video Upscaler`

Example workflows:

- `example_workflows/SeedVR2_simple_image_upscale.json`
- `example_workflows/SeedVR2_HD_video_upscale.json`
- `example_workflows/SeedVR2_4K_image_upscale.json`

Preview images are also included alongside those workflow files.

## Installation

### ComfyUI custom node

Clone or copy this repository into your ComfyUI custom nodes directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/itontonpi/seedvr2.git
```

Install dependencies in the same Python environment used by ComfyUI:

```bash
pip install -r requirements.txt
```

Restart ComfyUI after installation.

### Standalone CLI

Install the same dependencies, then run:

```bash
python inference_cli.py --help
```

## Models

Known models are defined in `src/utils/model_registry.py`. The default set includes:

- `seedvr2_ema_3b_fp8_e4m3fn.safetensors`
- `seedvr2_ema_3b_fp16.safetensors`
- `seedvr2_ema_7b_fp16.safetensors`
- 3B and 7B GGUF variants
- sharp 7B variants
- `ema_vae_fp16.safetensors`

By default, models are searched under `models/SEEDVR2`. In ComfyUI, the code also registers the `seedvr2` model type and can discover extra paths exposed through ComfyUI model folders.

## CLI Usage

Basic examples:

```bash
python inference_cli.py input.mp4 --resolution 1080
python inference_cli.py input.png --resolution 2048
python inference_cli.py media_folder/ --cache_dit --cache_vae --dit_offload_device cpu --vae_offload_device cpu
```

Selected capabilities:

- Single image input
- Single video input
- Directory batch processing
- Streaming mode with `--chunk_size`
- Multi-GPU execution with `--cuda_device`
- FFmpeg backend with `--video_backend ffmpeg`
- 10-bit FFmpeg output with `--10bit`

Use `python inference_cli.py --help` for the full flag list.

## Performance Notes

- `--tensor_offload_device cpu` is the default CLI setting and the safest option for long videos
- `--blocks_to_swap` and `--swap_io_components` reduce VRAM pressure for DiT inference
- VAE tiling is useful for high output resolutions
- `torch.compile` support is exposed in both the CLI and ComfyUI nodes, but it depends on your PyTorch and backend setup
- Flash Attention and SageAttention are optional backends; SDPA remains the baseline fallback

## Limitations

- The CLI is built around CUDA or Apple MPS paths; behavior on pure CPU setups is not a primary target
- FFmpeg output requires `ffmpeg` to be installed and available on `PATH`
- BlockSwap is documented as unavailable on macOS
- Batch sizing is tuned around the model's `4n+1` cadence, even though the pipeline pads or adjusts edge cases internally
- Supported model file extensions are `.safetensors` and `.gguf`

## Development Notes

- Package version is defined in `src/utils/constants.py`
- The package exports the ComfyUI entrypoint from `__init__.py`
- The standalone interface lives in `inference_cli.py`

## License

Apache 2.0. See `LICENSE`.
