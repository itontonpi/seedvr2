"""
Staged parallel batch processing for SeedVR2.

This module implements a correctness-first staged pipeline with explicit phases:

CPU prep pool -> H2D transfer stream -> GPU compute stream -> D2H copy stream -> CPU post pool -> ordered commit

It keeps a single GPU compute lane, but overlaps surrounding stages so the GPU
spends less time waiting on CPU preparation and CPU postprocessing.
"""

from concurrent.futures import Future, ThreadPoolExecutor
from time import perf_counter
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from .alpha_upscaling import process_alpha_for_batch
from .generation_phases import _apply_4n1_padding, _prepare_video_batch
from .generation_utils import (
    blend_overlapping_frames,
    calculate_optimal_batch_params,
    check_interrupt,
    setup_video_transform,
)
from .pipeline import _pipeline_cleanup, _setup_models
from ..common.seed import set_seed
from ..optimization.memory_manager import cleanup_dit, cleanup_text_embeddings, cleanup_vae
from ..optimization.performance import (
    optimized_sample_to_image_format,
    optimized_single_video_rearrange,
    optimized_video_rearrange,
)
from ..utils.color_fix import (
    adaptive_instance_normalization,
    hsv_saturation_histogram_match,
    lab_color_transfer,
    wavelet_adaptive_color_correction,
    wavelet_reconstruction,
)


def _maybe_pin(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if tensor.device.type == "cpu" and not tensor.is_pinned():
        return tensor.pin_memory()
    return tensor


def _prepare_batch_cpu(
    batch_idx: int,
    start_idx: int,
    end_idx: int,
    images: torch.Tensor,
    ctx: Dict[str, Any],
    batch_size: int,
    uniform_batch_size: bool,
    input_noise_scale: float,
    color_correction: str,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any], Dict[str, float]]:
    prep_start = perf_counter()

    current_frames = end_idx - start_idx
    is_uniform_padding = uniform_batch_size and current_frames < batch_size
    ori_length = current_frames
    uniform_padding = batch_size - current_frames if is_uniform_padding else 0

    video = _prepare_video_batch(
        images=images,
        start_idx=start_idx,
        end_idx=end_idx,
        uniform_padding=uniform_padding,
        debug=None,
        log_info=False,
    )

    padded_frames = batch_size if is_uniform_padding else current_frames
    video = _apply_4n1_padding(video)

    alpha_channel = None
    rgb_original = None
    if ctx.get("is_rgba", False):
        alpha_channel = video[:, 3:4, :, :].clone()
        rgb_original = video[:, :3, :, :].clone()
        rgb_video = video[:, :3, :, :]
    else:
        rgb_video = video

    transformed_video = ctx["video_transform"](rgb_video)

    if input_noise_scale > 0:
        noise = torch.randn_like(transformed_video)
        noise = noise * 0.05
        blend_factor = input_noise_scale * 0.5
        transformed_video = transformed_video * (1 - blend_factor) + (transformed_video + noise) * blend_factor

    cc_input = None
    if color_correction != "none":
        cc_input = optimized_single_video_rearrange(transformed_video.clone())
        actual_overlap = ctx.get("actual_temporal_overlap", 0)
        if batch_idx > 0 and actual_overlap > 0:
            cc_input = cc_input[actual_overlap:]

        if "true_target_dims" in ctx:
            th, tw = ctx["true_target_dims"]
            if cc_input.shape[-2] != th or cc_input.shape[-1] != tw:
                cc_input = cc_input[:, :, :th, :tw]

    metadata = {
        "batch_idx": batch_idx,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "ori_length": ori_length,
        "padded_frames": padded_frames,
        "alpha_channel": alpha_channel,
        "rgb_original": rgb_original,
    }

    stage_stats = {
        "prep_cpu": perf_counter() - prep_start,
        "prep_ready_at": perf_counter(),
    }
    return transformed_video, cc_input, metadata, stage_stats


def _postprocess_cpu(
    batch_idx: int,
    sample_cpu: torch.Tensor,
    cc_input: Optional[torch.Tensor],
    metadata: Dict[str, Any],
    ctx: Dict[str, Any],
    color_correction: str,
    debug: "Debug",
    copy_event: Optional[Any],
    stage_stats: Dict[str, float],
) -> Tuple[int, torch.Tensor, Dict[str, Any], Dict[str, float]]:
    post_wait_start = perf_counter()
    if copy_event is not None:
        copy_event.synchronize()
    stage_stats["d2h_complete_wait"] = perf_counter() - post_wait_start

    post_start = perf_counter()
    sample = sample_cpu

    if color_correction != "none" and cc_input is not None:
        has_alpha = ctx.get("is_rgba", False)
        alpha_ch = None

        if has_alpha and sample.shape[1] == 4:
            alpha_ch = sample[:, 3:4, :, :]
            sample = sample[:, :3, :, :]

        if cc_input.shape[0] > sample.shape[0]:
            cc_input = cc_input[:sample.shape[0]]

        if color_correction == "lab":
            sample = lab_color_transfer(sample, cc_input, debug, luminance_weight=0.8)
        elif color_correction == "wavelet_adaptive":
            sample = wavelet_adaptive_color_correction(sample, cc_input, debug)
        elif color_correction == "wavelet":
            sample = wavelet_reconstruction(sample, cc_input, debug)
        elif color_correction == "hsv":
            sample = hsv_saturation_histogram_match(sample, cc_input, debug)
        elif color_correction == "adain":
            sample = adaptive_instance_normalization(sample, cc_input)

        if has_alpha and alpha_ch is not None:
            sample = torch.cat([sample, alpha_ch], dim=1)

    sample = optimized_sample_to_image_format(sample)

    if ctx.get("is_rgba", False) and sample.shape[-1] == 4:
        rgb_ch = sample[..., :3]
        alpha_ch = sample[..., 3:4]
        rgb_ch.clamp_(-1, 1).mul_(0.5).add_(0.5)
        sample = torch.cat([rgb_ch, alpha_ch], dim=-1)
    else:
        sample.clamp_(-1, 1).mul_(0.5).add_(0.5)

    stage_stats["post_cpu"] = perf_counter() - post_start
    stage_stats["post_done_at"] = perf_counter()
    return batch_idx, sample, metadata, stage_stats


def _commit_batch(
    batch_idx: int,
    sample_thwc: torch.Tensor,
    metadata: Dict[str, Any],
    ctx: Dict[str, Any],
    temporal_overlap: int,
    current_write_idx: int,
) -> Tuple[int, int, int]:
    batch_frames = sample_thwc.shape[0]

    if batch_idx == 0 or temporal_overlap == 0:
        write_start = current_write_idx
        write_end = current_write_idx + batch_frames
        ctx["final_video"][write_start:write_end] = sample_thwc.to(ctx["final_video"].dtype)
        return write_end, write_start, write_end

    if temporal_overlap < batch_frames and current_write_idx >= temporal_overlap:
        prev_tail = ctx["final_video"][current_write_idx - temporal_overlap:current_write_idx]
        cur_head = sample_thwc[:temporal_overlap].to(prev_tail.device)
        blended = blend_overlapping_frames(prev_tail, cur_head, temporal_overlap)
        ctx["final_video"][current_write_idx - temporal_overlap:current_write_idx] = blended
        sample_thwc = sample_thwc[temporal_overlap:]
        batch_frames = sample_thwc.shape[0]

    write_start = current_write_idx
    write_end = current_write_idx + batch_frames
    ctx["final_video"][write_start:write_end] = sample_thwc.to(ctx["final_video"].dtype)
    return write_end, write_start, write_end


def process_batches_staged(
    runner: "VideoDiffusionInfer",
    ctx: Dict[str, Any],
    images: torch.Tensor,
    debug: "Debug",
    batch_size: int = 5,
    uniform_batch_size: bool = False,
    seed: int = 42,
    temporal_overlap: int = 0,
    resolution: int = 1080,
    max_resolution: int = 0,
    input_noise_scale: float = 0.0,
    latent_noise_scale: float = 0.0,
    color_correction: str = "lab",
    prepend_frames: int = 0,
    dit_cache: bool = False,
    vae_cache: bool = False,
    progress_callback: Optional[Callable[[int, int, int, str], None]] = None,
    prep_workers: int = 2,
    post_workers: int = 2,
    stage_depth: int = 3,
) -> Dict[str, Any]:
    debug.log("", category="none", force=True)
    debug.log("━━━━━━━━ Staged Pipeline: Processing ━━━━━━━━", category="none", force=True)
    debug.start_timer("staged_pipeline_total")

    if images is None:
        raise ValueError("Images must be provided")
    if ctx is None:
        raise ValueError("Generation context must be provided")

    ctx["input_images"] = images
    total_frames = ctx.get("total_frames", len(images))
    if "total_frames" not in ctx:
        ctx["total_frames"] = total_frames
    if total_frames == 0:
        raise ValueError("No frames to process")

    if "true_target_dims" not in ctx:
        sample_frame = images[0].permute(2, 0, 1).unsqueeze(0)
        setup_video_transform(ctx, resolution, max_resolution, debug, sample_frame)
        del sample_frame
    else:
        setup_video_transform(ctx, resolution, max_resolution, debug)

    ctx["is_rgba"] = images[0].shape[-1] == 4

    batch_params = calculate_optimal_batch_params(total_frames, batch_size, temporal_overlap)
    if batch_params["best_batch"] != batch_size and batch_params["best_batch"] <= total_frames:
        debug.log("", category="none", force=True)
        debug.log(
            f"Tip: For {total_frames} frames, batch_size={batch_params['best_batch']} matches video length optimally",
            category="tip",
            force=True,
        )
        debug.log("", category="none", force=True)

    step = batch_size - temporal_overlap if temporal_overlap > 0 else batch_size
    if step <= 0:
        step = batch_size
        temporal_overlap = 0
        debug.log("temporal_overlap >= batch_size, resetting to 0", level="WARNING", category="setup", force=True)

    ctx["actual_temporal_overlap"] = temporal_overlap

    batch_schedule = []
    for idx in range(0, total_frames, step):
        if idx == 0:
            start_idx = 0
            end_idx = min(batch_size, total_frames)
        else:
            start_idx = idx
            end_idx = min(start_idx + batch_size, total_frames)
            if end_idx - start_idx <= temporal_overlap:
                break
        batch_schedule.append((start_idx, end_idx))

    num_batches = len(batch_schedule)
    debug.log(f"Staged pipeline: {num_batches} batches, {total_frames} frames", category="generation", force=True)

    _setup_models(runner, ctx, debug, seed)

    true_h, true_w = ctx["true_target_dims"]
    channels = 4 if ctx.get("is_rgba", False) else 3
    required_gb = (total_frames * true_h * true_w * channels * 2) / (1024**3)
    debug.log(
        f"Pre-allocating output: {total_frames} frames, {true_w}x{true_h}px ({required_gb:.2f}GB)",
        category="setup",
        force=True,
    )
    ctx["final_video"] = torch.empty((total_frames, true_h, true_w, channels), dtype=ctx["compute_dtype"], device="cpu")

    gpu_device = ctx["vae_device"]
    use_cuda_streams = gpu_device.type == "cuda"
    transfer_stream = torch.cuda.Stream(device=gpu_device) if use_cuda_streams else None
    compute_stream = torch.cuda.Stream(device=gpu_device) if use_cuda_streams else None
    copy_stream = torch.cuda.Stream(device=gpu_device) if use_cuda_streams else None

    prep_workers = max(1, prep_workers)
    post_workers = max(1, post_workers)
    stage_depth = max(1, stage_depth)

    prep_futures: Dict[int, Future] = {}
    post_futures: Dict[int, Future] = {}
    stage_summaries = []
    next_commit_idx = 0
    current_write_idx = 0
    noise_buffer = None
    aug_noise_buffer = None

    def submit_prep(executor: ThreadPoolExecutor, batch_idx: int) -> None:
        if batch_idx in prep_futures or batch_idx >= num_batches:
            return
        start_idx, end_idx = batch_schedule[batch_idx]
        prep_futures[batch_idx] = executor.submit(
            _prepare_batch_cpu,
            batch_idx,
            start_idx,
            end_idx,
            images,
            ctx,
            batch_size,
            uniform_batch_size,
            input_noise_scale,
            color_correction,
        )

    def drain_commits(block: bool = False) -> None:
        nonlocal next_commit_idx, current_write_idx
        while next_commit_idx in post_futures:
            future = post_futures[next_commit_idx]
            if not block and not future.done():
                break

            commit_wait_start = perf_counter()
            batch_idx, sample_thwc, metadata, stage_stats = future.result()
            stage_stats["commit_wait"] = perf_counter() - commit_wait_start
            current_write_idx, write_start, write_end = _commit_batch(
                batch_idx, sample_thwc, metadata, ctx, temporal_overlap, current_write_idx
            )
            stage_stats["committed_frames"] = int(write_end - write_start)
            stage_stats["commit_done_at"] = perf_counter()
            stage_summaries.append(stage_stats)

            if debug.enabled:
                debug.log(
                    "Staged batch "
                    f"{batch_idx+1}/{num_batches}: "
                    f"prep_wait={stage_stats.get('prep_wait', 0.0):.3f}s, "
                    f"prep_cpu={stage_stats.get('prep_cpu', 0.0):.3f}s, "
                    f"h2d_enqueue={stage_stats.get('h2d_enqueue', 0.0):.3f}s, "
                    f"h2d_wait={stage_stats.get('h2d_wait', 0.0):.3f}s, "
                    f"gpu_compute={stage_stats.get('gpu_compute', 0.0):.3f}s, "
                    f"d2h_enqueue={stage_stats.get('d2h_enqueue', 0.0):.3f}s, "
                    f"d2h_wait={stage_stats.get('d2h_complete_wait', 0.0):.3f}s, "
                    f"post_cpu={stage_stats.get('post_cpu', 0.0):.3f}s, "
                    f"commit_wait={stage_stats.get('commit_wait', 0.0):.3f}s",
                    category="generation",
                    force=True,
                )

            del post_futures[next_commit_idx]
            next_commit_idx += 1

    try:
        with ThreadPoolExecutor(max_workers=prep_workers, thread_name_prefix="seedvr2-staged-prep") as prep_executor, \
             ThreadPoolExecutor(max_workers=post_workers, thread_name_prefix="seedvr2-staged-post") as post_executor:
            for idx in range(min(stage_depth, num_batches)):
                submit_prep(prep_executor, idx)

            for batch_idx in range(num_batches):
                check_interrupt(ctx)
                submit_prep(prep_executor, batch_idx + stage_depth)

                prep_wait_start = perf_counter()
                transformed_video, cc_input, metadata, stage_stats = prep_futures.pop(batch_idx).result()
                stage_stats["prep_wait"] = perf_counter() - prep_wait_start
                ori_length = metadata["ori_length"]

                debug.start_timer(f"staged_pipeline_batch_{batch_idx+1}")

                transformed_video = _maybe_pin(transformed_video)
                alpha_channel = _maybe_pin(metadata.get("alpha_channel"))
                rgb_original = _maybe_pin(metadata.get("rgb_original"))

                h2d_enqueue_start = perf_counter()
                if use_cuda_streams:
                    with torch.cuda.stream(transfer_stream):
                        transformed_video_gpu = transformed_video.to(gpu_device, dtype=ctx["compute_dtype"], non_blocking=True)
                    h2d_ready = torch.cuda.Event()
                    transfer_stream.record_event(h2d_ready)
                else:
                    transformed_video_gpu = transformed_video.to(gpu_device, dtype=ctx["compute_dtype"])
                    h2d_ready = None
                stage_stats["h2d_enqueue"] = perf_counter() - h2d_enqueue_start

                compute_wait_start = perf_counter()
                if use_cuda_streams:
                    with torch.cuda.stream(compute_stream):
                        compute_stream.wait_event(h2d_ready)
                    torch.cuda.set_stream(compute_stream)
                stage_stats["h2d_wait"] = perf_counter() - compute_wait_start

                gpu_compute_start = perf_counter()
                cond_latents = runner.vae_encode([transformed_video_gpu])
                latent = cond_latents[0]
                del transformed_video_gpu, cond_latents

                if latent.device != ctx["dit_device"]:
                    latent = latent.to(ctx["dit_device"], dtype=ctx["compute_dtype"], non_blocking=use_cuda_streams)

                set_seed(seed)
                if noise_buffer is None or noise_buffer.shape != latent.shape:
                    noise_buffer = torch.empty_like(latent, dtype=ctx["compute_dtype"])
                    aug_noise_buffer = torch.empty_like(latent, dtype=ctx["compute_dtype"])
                noise_buffer.normal_()
                aug_noise_buffer.normal_()

                noises = [noise_buffer]
                aug_noises = [noise_buffer * 0.1 + aug_noise_buffer * 0.05]

                def _add_noise(x, aug_noise):
                    if latent_noise_scale == 0.0:
                        return x
                    t = torch.tensor([1000.0], device=ctx["dit_device"], dtype=ctx["compute_dtype"]) * latent_noise_scale
                    shape = torch.tensor(x.shape[1:], device=ctx["dit_device"])[None]
                    t = runner.timestep_transform(t, shape)
                    x = runner.schedule.forward(x, aug_noise, t)
                    del t, shape
                    return x

                condition = runner.get_condition(
                    noises[0],
                    task="sr",
                    latent_blur=_add_noise(latent, aug_noises[0]),
                )

                dit_model = runner.dit.dit_model if hasattr(runner.dit, "dit_model") else runner.dit
                try:
                    dit_dtype = next(dit_model.parameters()).dtype
                except StopIteration:
                    dit_dtype = ctx["compute_dtype"]

                with torch.no_grad():
                    if dit_dtype != ctx["compute_dtype"] and ctx["dit_device"].type != "mps":
                        with torch.autocast(ctx["dit_device"].type, ctx["compute_dtype"], enabled=True):
                            upscaled_latents = runner.inference(noises=noises, conditions=[condition], **ctx["text_embeds"])
                    else:
                        upscaled_latents = runner.inference(noises=noises, conditions=[condition], **ctx["text_embeds"])

                upscaled_latent = upscaled_latents[0]
                del latent, noises, aug_noises, condition, upscaled_latents

                if upscaled_latent.device != ctx["vae_device"]:
                    upscaled_latent = upscaled_latent.to(ctx["vae_device"], dtype=ctx["compute_dtype"], non_blocking=use_cuda_streams)

                decoded = runner.vae_decode([upscaled_latent])
                del upscaled_latent

                samples = optimized_video_rearrange(decoded)
                sample = samples[0]
                del decoded, samples

                if ori_length < sample.shape[0]:
                    sample = sample[:ori_length]

                current_h, current_w = sample.shape[-2:]
                if current_h != true_h or current_w != true_w:
                    sample = sample[:, :, :true_h, :true_w]

                if alpha_channel is not None and ctx.get("is_rgba", False):
                    alpha_channel_gpu = alpha_channel.to(gpu_device, non_blocking=use_cuda_streams)
                    rgb_original_gpu = rgb_original.to(gpu_device, non_blocking=use_cuda_streams) if rgb_original is not None else None
                    rgb_tchw = sample[:, :3, :, :]

                    processed = process_alpha_for_batch(
                        rgb_samples=[rgb_tchw],
                        alpha_original=alpha_channel_gpu,
                        rgb_original=rgb_original_gpu,
                        device=gpu_device,
                        compute_dtype=ctx["compute_dtype"],
                        debug=debug,
                    )
                    sample = processed[0]

                stage_stats["gpu_compute"] = perf_counter() - gpu_compute_start

                d2h_enqueue_start = perf_counter()
                if use_cuda_streams:
                    sample_cpu = torch.empty_like(sample, device="cpu", pin_memory=True)
                    with torch.cuda.stream(copy_stream):
                        copy_stream.wait_stream(compute_stream)
                        sample_cpu.copy_(sample, non_blocking=True)
                    copy_event = torch.cuda.Event()
                    copy_stream.record_event(copy_event)
                else:
                    sample_cpu = sample.cpu()
                    copy_event = None
                stage_stats["d2h_enqueue"] = perf_counter() - d2h_enqueue_start

                post_futures[batch_idx] = post_executor.submit(
                    _postprocess_cpu,
                    batch_idx,
                    sample_cpu,
                    cc_input,
                    metadata,
                    ctx,
                    color_correction,
                    debug,
                    copy_event,
                    stage_stats,
                )

                if use_cuda_streams:
                    torch.cuda.set_stream(torch.cuda.default_stream(gpu_device))

                drain_commits(block=False)
                while len(post_futures) >= stage_depth:
                    drain_commits(block=True)

                debug.end_timer(f"staged_pipeline_batch_{batch_idx+1}", f"Staged pipeline batch {batch_idx+1}")
                if progress_callback:
                    progress_callback(
                        int(((batch_idx + 1) / num_batches) * 100),
                        100,
                        int(sample.shape[0]),
                        f"Staged pipeline batch {batch_idx+1}/{num_batches}",
                    )

            while post_futures:
                drain_commits(block=True)

    finally:
        cleanup_dit(runner=runner, debug=debug, cache_model=dit_cache)
        cleanup_vae(runner=runner, debug=debug, cache_model=vae_cache)
        cleanup_text_embeddings(ctx, debug)

    if debug.enabled and stage_summaries:
        avg = lambda key: sum(x.get(key, 0.0) for x in stage_summaries) / len(stage_summaries)
        debug.log(
            "Staged pipeline averages: "
            f"prep_wait={avg('prep_wait'):.3f}s, "
            f"prep_cpu={avg('prep_cpu'):.3f}s, "
            f"h2d_enqueue={avg('h2d_enqueue'):.3f}s, "
            f"h2d_wait={avg('h2d_wait'):.3f}s, "
            f"gpu_compute={avg('gpu_compute'):.3f}s, "
            f"d2h_enqueue={avg('d2h_enqueue'):.3f}s, "
            f"d2h_wait={avg('d2h_complete_wait'):.3f}s, "
            f"post_cpu={avg('post_cpu'):.3f}s, "
            f"commit_wait={avg('commit_wait'):.3f}s",
            category="generation",
            force=True,
        )

    if prepend_frames > 0 and prepend_frames < ctx["final_video"].shape[0]:
        debug.log(f"Removing {prepend_frames} prepended frames from output", category="video", force=True)
        ctx["final_video"] = ctx["final_video"][prepend_frames:]

    final_shape = ctx["final_video"].shape
    tf, hf, wf, cf = final_shape
    channels_str = "RGBA" if cf == 4 else "RGB"
    debug.log(f"Output assembled: {tf} frames, {wf}x{hf}px, {channels_str}", category="generation", force=True)

    _pipeline_cleanup(ctx)
    debug.end_timer("staged_pipeline_total", "Staged pipeline processing complete", show_breakdown=True)
    return ctx
