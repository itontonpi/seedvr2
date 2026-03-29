"""
Pipelined Batch Processing for SeedVR2

Processes each batch through all 4 phases (encode→upscale→decode→postprocess)
as a unit, using 3 threads connected by queues for maximum GPU utilization:

  CPU Prep Thread  →  GPU Main Thread  →  CPU Post Thread
  (batch prep)        (enc→ups→dec)        (color correction)

Key advantages over sequential all-batches-per-phase:
  - Zero model swapping (both VAE and DiT stay on GPU)
  - CPU batch prep overlaps with GPU compute
  - CPU color correction overlaps with GPU compute
  - GPU utilization stays near 100%

Usage:
    ctx = process_batches_pipelined(runner, ctx, images, debug, ...)
    # Replaces: encode_all → upscale_all → decode_all → postprocess_all
"""

import threading
from queue import Queue
import torch
from typing import Dict, List, Optional, Tuple, Any, Callable

from .generation_utils import (
    setup_video_transform,
    check_interrupt,
    ensure_precision_initialized,
    load_text_embeddings,
    blend_overlapping_frames,
    calculate_optimal_batch_params,
    script_directory
)
from .generation_phases import (
    _prepare_video_batch,
    _apply_4n1_padding,
    _reconstruct_and_transform_batch,
)
from .model_configuration import apply_model_specific_config
from .model_loader import materialize_model
from .alpha_upscaling import process_alpha_for_batch
from .infer import VideoDiffusionInfer
from ..common.seed import set_seed
from ..optimization.memory_manager import (
    cleanup_dit,
    cleanup_vae,
    cleanup_text_embeddings,
    manage_tensor,
    manage_model_device,
    release_tensor_memory,
    release_tensor_collection
)
from ..optimization.performance import (
    optimized_video_rearrange,
    optimized_single_video_rearrange,
    optimized_sample_to_image_format
)
from ..utils.color_fix import (
    lab_color_transfer,
    wavelet_adaptive_color_correction,
    hsv_saturation_histogram_match,
    wavelet_reconstruction,
    adaptive_instance_normalization
)


# Sentinel value to signal end of queue
_SENTINEL = None

# Queue sizes for backpressure (double-buffer)
_PREP_QUEUE_SIZE = 2
_POST_QUEUE_SIZE = 2


def process_batches_pipelined(
    runner: 'VideoDiffusionInfer',
    ctx: Dict[str, Any],
    images: torch.Tensor,
    debug: 'Debug',
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
) -> Dict[str, Any]:
    """
    Pipeline: process each batch through all 4 phases with GPU-CPU overlap.
    
    Both VAE and DiT stay on GPU simultaneously. Each batch flows through
    encode → upscale → decode on GPU, while CPU threads handle batch prep
    and color correction concurrently.
    
    Args:
        runner: VideoDiffusionInfer instance with loaded models
        ctx: Generation context from setup_generation_context
        images: Input frames tensor [T, H, W, C] range [0,1]
        debug: Debug instance for logging
        batch_size: Frames per batch (4n+1 format)
        uniform_batch_size: Pad final batch to match batch_size
        seed: Random seed for reproducible generation
        temporal_overlap: Overlapping frames between batches
        resolution: Target resolution for shortest edge
        max_resolution: Maximum resolution for any edge (0 = no limit)
        input_noise_scale: Input noise injection scale (0.0-1.0)
        latent_noise_scale: Latent noise injection scale (0.0-1.0)
        color_correction: Color correction method
        prepend_frames: Number of prepended frames to remove
        dit_cache: Keep DiT model after processing
        vae_cache: Keep VAE model after processing
        progress_callback: Optional progress callback
    
    Returns:
        Updated context with final_video tensor
    """
    debug.log("", category="none", force=True)
    debug.log("━━━━━━━━ Pipeline Mode: Processing ━━━━━━━━", category="none", force=True)
    debug.start_timer("pipeline_total")
    
    # ─── Validate inputs ───
    if images is None:
        raise ValueError("Images must be provided")
    if ctx is None:
        raise ValueError("Generation context must be provided")
    
    # Store input images for color correction
    ctx['input_images'] = images
    
    total_frames = ctx.get('total_frames', len(images))
    if 'total_frames' not in ctx:
        ctx['total_frames'] = total_frames
    
    if total_frames == 0:
        raise ValueError("No frames to process")
    
    # ─── Setup video transform ───
    if 'true_target_dims' not in ctx:
        sample_frame = images[0].permute(2, 0, 1).unsqueeze(0)
        setup_video_transform(ctx, resolution, max_resolution, debug, sample_frame)
        del sample_frame
    else:
        setup_video_transform(ctx, resolution, max_resolution, debug)
    
    ctx['is_rgba'] = images[0].shape[-1] == 4
    
    # Display batch optimization tip
    if total_frames > 0:
        batch_params = calculate_optimal_batch_params(total_frames, batch_size, temporal_overlap)
        if batch_params['best_batch'] != batch_size and batch_params['best_batch'] <= total_frames:
            debug.log("", category="none", force=True)
            debug.log(f"Tip: For {total_frames} frames, batch_size={batch_params['best_batch']} matches video length optimally", 
                     category="tip", force=True)
            debug.log("", category="none", force=True)
    
    # ─── Calculate batching parameters ───
    step = batch_size - temporal_overlap if temporal_overlap > 0 else batch_size
    if step <= 0:
        step = batch_size
        temporal_overlap = 0
        debug.log("temporal_overlap >= batch_size, resetting to 0", level="WARNING", category="setup", force=True)
    
    ctx['actual_temporal_overlap'] = temporal_overlap
    
    # Build batch schedule: list of (start_idx, end_idx)
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
    debug.log(f"Pipeline: {num_batches} batches, {total_frames} frames", category="generation", force=True)
    
    # ─── Materialize and prepare models ───
    _setup_models(runner, ctx, debug, seed)
    
    # ─── Pre-allocate output tensor ───
    true_h, true_w = ctx['true_target_dims']
    C = 4 if ctx.get('is_rgba', False) else 3
    required_gb = (total_frames * true_h * true_w * C * 2) / (1024**3)
    debug.log(f"Pre-allocating output: {total_frames} frames, {true_w}x{true_h}px ({required_gb:.2f}GB)", 
              category="setup", force=True)
    ctx['final_video'] = torch.empty(
        (total_frames, true_h, true_w, C), 
        dtype=ctx['compute_dtype'], 
        device='cpu'
    )
    
    # ─── Create queues ───
    prep_queue = Queue(maxsize=_PREP_QUEUE_SIZE)
    post_queue = Queue(maxsize=_POST_QUEUE_SIZE)
    
    # Thread error container
    thread_errors = []
    thread_error_lock = threading.Lock()
    
    # ─── Define prep worker ───
    def prep_worker():
        """Pre-compute batch data on CPU → feed to GPU via queue."""
        try:
            for batch_idx, (start_idx, end_idx) in enumerate(batch_schedule):
                current_frames = end_idx - start_idx
                is_uniform_padding = uniform_batch_size and current_frames < batch_size
                ori_length = current_frames
                uniform_padding = batch_size - current_frames if is_uniform_padding else 0
                
                # Prepare video batch (CPU: slice, permute, pad)
                video = _prepare_video_batch(
                    images=images,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    uniform_padding=uniform_padding,
                    debug=None,  # No logging from background thread
                    log_info=False
                )
                
                padded_frames = batch_size if is_uniform_padding else current_frames
                
                # Apply 4n+1 temporal padding
                video = _apply_4n1_padding(video)
                
                # Extract RGBA alpha if needed
                alpha_channel = None
                rgb_original = None
                if ctx.get('is_rgba', False):
                    alpha_channel = video[:, 3:4, :, :].clone()
                    rgb_original = video[:, :3, :, :].clone()
                    rgb_video = video[:, :3, :, :]
                else:
                    rgb_video = video
                
                # Apply video transform (CPU: resize + normalize)
                transformed_video = ctx['video_transform'](rgb_video)
                
                # Apply input noise if requested
                if input_noise_scale > 0:
                    noise = torch.randn_like(transformed_video)
                    noise = noise * 0.05
                    blend_factor = input_noise_scale * 0.5
                    transformed_video = transformed_video * (1 - blend_factor) + (transformed_video + noise) * blend_factor
                    del noise
                
                # Pre-compute color correction input (same transform applied to input)
                cc_input = None
                if color_correction != "none":
                    # Clone the transformed video as color correction reference
                    # Rearrange from TCHW to CTHW for color correction
                    cc_input = optimized_single_video_rearrange(transformed_video.clone())
                    
                    # Apply temporal overlap trimming for color correction input
                    actual_overlap = ctx.get('actual_temporal_overlap', 0)
                    if batch_idx > 0 and actual_overlap > 0:
                        cc_input = cc_input[actual_overlap:]
                    
                    # Trim spatial to true target
                    if 'true_target_dims' in ctx:
                        th, tw = ctx['true_target_dims']
                        if cc_input.shape[-2] != th or cc_input.shape[-1] != tw:
                            cc_input = cc_input[:, :, :th, :tw]
                
                del video
                
                # Package metadata
                metadata = {
                    'batch_idx': batch_idx,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'ori_length': ori_length,
                    'padded_frames': padded_frames,
                    'alpha_channel': alpha_channel,
                    'rgb_original': rgb_original,
                }
                
                prep_queue.put((transformed_video, cc_input, metadata))
            
            # Signal end
            prep_queue.put(_SENTINEL)
            
        except Exception as e:
            with thread_error_lock:
                thread_errors.append(('prep', e))
            prep_queue.put(_SENTINEL)
    
    # ─── Define post worker ───
    def post_worker():
        """Color correction + normalization on CPU, concurrent with GPU."""
        try:
            while True:
                item = post_queue.get()
                if item is _SENTINEL:
                    break
                
                sample, cc_input, write_start, write_end, metadata = item
                batch_idx = metadata['batch_idx']
                
                # sample is [T, C, H, W] from GPU, now on CPU
                # Apply color correction
                if color_correction != "none" and cc_input is not None:
                    has_alpha = ctx.get('is_rgba', False)
                    alpha_ch = None
                    
                    if has_alpha and sample.shape[1] == 4:
                        alpha_ch = sample[:, 3:4, :, :]
                        sample = sample[:, :3, :, :]
                    
                    # Trim cc_input to match sample length
                    if cc_input.shape[0] > sample.shape[0]:
                        cc_input = cc_input[:sample.shape[0]]
                    
                    # Apply selected color correction method
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
                    
                    del cc_input
                    
                    if has_alpha and alpha_ch is not None:
                        sample = torch.cat([sample, alpha_ch], dim=1)
                
                # Convert [T, C, H, W] → [T, H, W, C]
                sample = optimized_sample_to_image_format(sample)
                
                # Normalize [-1, 1] → [0, 1]
                if ctx.get('is_rgba', False) and sample.shape[-1] == 4:
                    rgb_ch = sample[..., :3]
                    alpha_ch = sample[..., 3:4]
                    rgb_ch.clamp_(-1, 1).mul_(0.5).add_(0.5)
                    sample = torch.cat([rgb_ch, alpha_ch], dim=-1)
                else:
                    sample.clamp_(-1, 1).mul_(0.5).add_(0.5)
                
                # Write to final_video
                ctx['final_video'][write_start:write_end] = sample.to(ctx['final_video'].device)
                
                del sample
                
        except Exception as e:
            with thread_error_lock:
                thread_errors.append(('post', e))
    
    # ─── Start background threads ───
    prep_thread = threading.Thread(target=prep_worker, name="pipeline-prep", daemon=True)
    post_thread = threading.Thread(target=post_worker, name="pipeline-post", daemon=True)
    prep_thread.start()
    post_thread.start()
    
    # ─── GPU Main Loop ───
    noise_buffer = None
    aug_noise_buffer = None
    current_write_idx = 0
    gpu_device = ctx['vae_device']
    
    try:
        batch_count = 0
        
        while True:
            # Check for prep thread errors
            with thread_error_lock:
                if thread_errors:
                    raise thread_errors[0][1]
            
            item = prep_queue.get()
            if item is _SENTINEL:
                break
            
            transformed_video, cc_input, metadata = item
            batch_idx = metadata['batch_idx']
            ori_length = metadata['ori_length']
            
            check_interrupt(ctx)
            
            debug.log(f"Pipeline batch {batch_idx+1}/{num_batches}", category="generation", force=True)
            debug.start_timer(f"pipeline_batch_{batch_idx+1}")
            
            # ── Phase 1: VAE Encode ──
            debug.start_timer(f"pipe_encode_{batch_idx+1}")
            
            # Move to GPU
            transformed_video = transformed_video.to(gpu_device, dtype=ctx['compute_dtype'], non_blocking=True)
            if gpu_device.type == 'cuda':
                torch.cuda.current_stream().synchronize()
            
            cond_latents = runner.vae_encode([transformed_video])
            latent = cond_latents[0]
            del transformed_video, cond_latents
            
            debug.end_timer(f"pipe_encode_{batch_idx+1}", f"  Encode batch {batch_idx+1}")
            
            # ── Phase 2: DiT Upscale ──
            debug.start_timer(f"pipe_upscale_{batch_idx+1}")
            
            # Ensure latent is on DiT device with correct dtype
            if latent.device != ctx['dit_device']:
                latent = latent.to(ctx['dit_device'], dtype=ctx['compute_dtype'], non_blocking=True)
                if ctx['dit_device'].type == 'cuda':
                    torch.cuda.current_stream().synchronize()
            
            set_seed(seed)
            
            # Reuse noise buffers
            if noise_buffer is None or noise_buffer.shape != latent.shape:
                noise_buffer = torch.empty_like(latent, dtype=ctx['compute_dtype'])
                aug_noise_buffer = torch.empty_like(latent, dtype=ctx['compute_dtype'])
            noise_buffer.normal_()
            aug_noise_buffer.normal_()
            
            noises = [noise_buffer]
            aug_noises = [noise_buffer * 0.1 + aug_noise_buffer * 0.05]
            
            # Latent noise conditioning
            def _add_noise(x, aug_noise):
                if latent_noise_scale == 0.0:
                    return x
                t = torch.tensor([1000.0], device=ctx['dit_device'], dtype=ctx['compute_dtype']) * latent_noise_scale
                shape = torch.tensor(x.shape[1:], device=ctx['dit_device'])[None]
                t = runner.timestep_transform(t, shape)
                x = runner.schedule.forward(x, aug_noise, t)
                del t, shape
                return x
            
            condition = runner.get_condition(
                noises[0], task="sr",
                latent_blur=_add_noise(latent, aug_noises[0]),
            )
            
            # Detect DiT dtype for autocast
            dit_model = runner.dit.dit_model if hasattr(runner.dit, 'dit_model') else runner.dit
            try:
                dit_dtype = next(dit_model.parameters()).dtype
            except StopIteration:
                dit_dtype = ctx['compute_dtype']
            
            with torch.no_grad():
                if dit_dtype != ctx['compute_dtype'] and ctx['dit_device'].type != 'mps':
                    with torch.autocast(ctx['dit_device'].type, ctx['compute_dtype'], enabled=True):
                        upscaled_latents = runner.inference(
                            noises=noises, conditions=[condition],
                            **ctx['text_embeds'],
                        )
                else:
                    upscaled_latents = runner.inference(
                        noises=noises, conditions=[condition],
                        **ctx['text_embeds'],
                    )
            
            upscaled_latent = upscaled_latents[0]
            del latent, noises, aug_noises, condition, upscaled_latents
            
            debug.end_timer(f"pipe_upscale_{batch_idx+1}", f"  Upscale batch {batch_idx+1}")
            
            # ── Phase 3: VAE Decode ──
            debug.start_timer(f"pipe_decode_{batch_idx+1}")
            
            # Move to VAE device if needed
            if upscaled_latent.device != ctx['vae_device']:
                upscaled_latent = upscaled_latent.to(ctx['vae_device'], dtype=ctx['compute_dtype'], non_blocking=True)
                if ctx['vae_device'].type == 'cuda':
                    torch.cuda.current_stream().synchronize()
            
            decoded = runner.vae_decode([upscaled_latent])
            del upscaled_latent
            
            # Rearrange decoded output
            samples = optimized_video_rearrange(decoded)
            sample = samples[0]
            del decoded, samples
            
            # Trim temporal padding
            if ori_length < sample.shape[0]:
                sample = sample[:ori_length]
            
            # Trim spatial padding to true target dimensions
            current_h, current_w = sample.shape[-2:]
            if current_h != true_h or current_w != true_w:
                sample = sample[:, :, :true_h, :true_w]
            
            debug.end_timer(f"pipe_decode_{batch_idx+1}", f"  Decode batch {batch_idx+1}")
            
            # ── Handle temporal overlap blending ──
            batch_frames = sample.shape[0]
            if batch_idx == 0 or temporal_overlap == 0:
                write_start = current_write_idx
                write_end = current_write_idx + batch_frames
            else:
                # For overlap blending, convert to THWC, blend, then back
                sample_thwc = optimized_sample_to_image_format(sample)
                
                if temporal_overlap < batch_frames and current_write_idx >= temporal_overlap:
                    prev_tail = ctx['final_video'][current_write_idx - temporal_overlap:current_write_idx]
                    cur_head = sample_thwc[:temporal_overlap].to(prev_tail.device)
                    
                    blended = blend_overlapping_frames(prev_tail, cur_head, temporal_overlap)
                    ctx['final_video'][current_write_idx - temporal_overlap:current_write_idx] = blended
                    
                    sample_thwc = sample_thwc[temporal_overlap:]
                    batch_frames = sample_thwc.shape[0]
                    del prev_tail, cur_head, blended
                
                # Convert back to TCHW for post-processing (color correction expects TCHW)
                sample = sample_thwc.permute(0, 3, 1, 2)
                del sample_thwc
                
                write_start = current_write_idx
                write_end = current_write_idx + batch_frames
            
            current_write_idx = write_end
            
            # ── Handle RGBA alpha ──
            alpha_channel = metadata.get('alpha_channel')
            if alpha_channel is not None and ctx.get('is_rgba', False):
                rgb_original = metadata.get('rgb_original')
                # Process alpha upscaling
                rgb_tchw = sample[:, :3, :, :]  # Already in TCHW
                
                processed = process_alpha_for_batch(
                    rgb_samples=[rgb_tchw.to(gpu_device)],
                    alpha_original=alpha_channel.to(gpu_device),
                    rgb_original=rgb_original.to(gpu_device) if rgb_original is not None else None,
                    device=gpu_device,
                    compute_dtype=ctx['compute_dtype'],
                    debug=debug
                )
                
                # Extract upscaled alpha
                alpha_up = processed[0][:, 3:4, :, :]  # [T, 1, H, W]
                alpha_thwc = alpha_up.permute(0, 2, 3, 1).cpu()  # [T, H, W, 1]
                
                # Trim to match write range
                frames_to_write = write_end - write_start
                if alpha_thwc.shape[0] > frames_to_write:
                    alpha_thwc = alpha_thwc[:frames_to_write]
                
                ctx['final_video'][write_start:write_end, :, :, 3:4] = alpha_thwc.to(ctx['final_video'].dtype)
                del processed, alpha_up, alpha_thwc, alpha_channel, rgb_original
            
            # ── Send to post worker for color correction ──
            # Move sample to CPU for post-processing thread
            sample_cpu = sample.cpu()
            del sample
            
            post_queue.put((sample_cpu, cc_input, write_start, write_end, metadata))
            
            debug.end_timer(f"pipeline_batch_{batch_idx+1}", f"Pipeline batch {batch_idx+1}")
            
            # Progress callback
            if progress_callback:
                # Map to overall progress (encode=20%, upscale=25%, decode=50%, post=5%)
                overall = (batch_idx + 1) / num_batches
                progress_callback(
                    int(overall * 100), 100,
                    batch_frames, f"Pipeline batch {batch_idx+1}/{num_batches}"
                )
            
            batch_count += 1
        
    except Exception as e:
        debug.log(f"Error in pipeline GPU loop: {e}", level="ERROR", category="error", force=True)
        # Drain queues to unblock threads
        post_queue.put(_SENTINEL)
        raise
    
    finally:
        # Signal post worker to finish
        post_queue.put(_SENTINEL)
        
        # Wait for threads to complete
        prep_thread.join(timeout=30)
        post_thread.join(timeout=30)
        
        # Cleanup models
        cleanup_dit(runner=runner, debug=debug, cache_model=dit_cache)
        cleanup_vae(runner=runner, debug=debug, cache_model=vae_cache)
        cleanup_text_embeddings(ctx, debug)
    
    # Check for thread errors
    with thread_error_lock:
        if thread_errors:
            phase, err = thread_errors[0]
            debug.log(f"Error in pipeline {phase} thread: {err}", level="ERROR", category="error", force=True)
            raise err
    
    # ─── Post-pipeline finalization ───
    # Remove prepended frames
    if prepend_frames > 0 and prepend_frames < ctx['final_video'].shape[0]:
        debug.log(f"Removing {prepend_frames} prepended frames from output", category="video", force=True)
        ctx['final_video'] = ctx['final_video'][prepend_frames:]
    
    # Log output info
    final_shape = ctx['final_video'].shape
    Tf, Hf, Wf, Cf = final_shape
    channels_str = "RGBA" if Cf == 4 else "RGB"
    debug.log(f"Output assembled: {Tf} frames, {Wf}x{Hf}px, {channels_str}", category="generation", force=True)
    
    # Cleanup
    _pipeline_cleanup(ctx)
    
    debug.end_timer("pipeline_total", "Pipeline processing complete", show_breakdown=True)
    
    return ctx


def _setup_models(runner, ctx, debug, seed):
    """Materialize and prepare both models on GPU before pipeline starts."""
    
    # ─── VAE Setup ───
    if runner.vae and next(runner.vae.parameters()).device.type == 'meta':
        materialize_model(runner, "vae", ctx['vae_device'], runner.config, debug)

    cache_context = ctx.get('cache_context')
    if cache_context and cache_context.get('vae_cache') and not cache_context.get('cached_vae'):
        runner.vae._model_name = cache_context['vae_model']
        cache_context['global_cache'].set_vae(
            {'node_id': cache_context['vae_id'], 'cache_model': True},
            runner.vae, cache_context['vae_model'], debug
        )
        cache_context['vae_newly_cached'] = True
    
    ensure_precision_initialized(ctx, runner, debug)
    
    manage_model_device(
        model=runner.vae, target_device=ctx['vae_device'],
        model_name="VAE", debug=debug, runner=runner
    )
    
    # ─── DiT Setup ───
    if runner.dit and next(runner.dit.parameters()).device.type == 'meta':
        materialize_model(runner, "dit", ctx['dit_device'], runner.config, debug)
    else:
        if getattr(runner, '_dit_config_needs_application', False):
            apply_model_specific_config(runner.dit, runner, runner.config, True, debug)

    if cache_context and cache_context.get('dit_cache') and not cache_context.get('cached_dit'):
        runner.dit._model_name = cache_context['dit_model']
        cache_context['global_cache'].set_dit(
            {'node_id': cache_context['dit_id'], 'cache_model': True},
            runner.dit, cache_context['dit_model'], debug
        )
        cache_context['dit_newly_cached'] = True

    if cache_context:
        dit_is_cached = cache_context.get('cached_dit') or cache_context.get('dit_newly_cached')
        vae_is_cached = cache_context.get('cached_vae') or cache_context.get('vae_newly_cached')
        if dit_is_cached and vae_is_cached:
            cache_context['global_cache'].set_runner(
                cache_context['dit_id'], cache_context['vae_id'],
                runner, debug
            )

    ensure_precision_initialized(ctx, runner, debug)
    
    # Configure diffusion (1-step distilled)
    runner.config.diffusion.cfg.scale = 1.0
    runner.config.diffusion.cfg.rescale = 0.0
    runner.config.diffusion.timesteps.sampling.steps = 1
    runner.configure_diffusion(device=ctx['dit_device'], dtype=ctx['compute_dtype'])
    
    manage_model_device(
        model=runner.dit, target_device=ctx['dit_device'],
        model_name="DiT", debug=debug, runner=runner
    )
    
    # ─── Text embeddings ───
    if ctx.get('text_embeds') is None:
        ctx['text_embeds'] = load_text_embeddings(
            script_directory, ctx['dit_device'], ctx['compute_dtype'], debug
        )
    
    # ─── VAE seed ───
    seed_vae = seed + 1000000
    set_seed(seed_vae)
    
    debug.log("Pipeline: Both VAE and DiT loaded on GPU", category="generation", force=True)
    debug.log_memory_state("After pipeline model setup", show_tensors=False, detailed_tensors=False)


def _pipeline_cleanup(ctx):
    """Clean up intermediate pipeline state from context."""
    cleanup_keys = [
        'all_latents', 'all_ori_lengths', 'all_alpha_channels', 'all_input_rgb',
        'all_upscaled_latents', 'batch_metadata', 'decode_batch_info',
        'total_padding_removed', 'true_target_dims', 'all_ori_lengths'
    ]
    
    for key in cleanup_keys:
        if key in ctx:
            val = ctx[key]
            if isinstance(val, list):
                release_tensor_collection(val)
            elif isinstance(val, torch.Tensor):
                release_tensor_memory(val)
            del ctx[key]
    
    if 'video_transform' in ctx and ctx['video_transform'] is not None:
        if hasattr(ctx['video_transform'], 'transforms'):
            for transform in ctx['video_transform'].transforms:
                for cache_attr in ['cache', '_cache']:
                    if hasattr(transform, cache_attr):
                        setattr(transform, cache_attr, None)
                if hasattr(transform, '__dict__'):
                    transform.__dict__.clear()
        del ctx['video_transform']
    
    if 'input_images' in ctx:
        release_tensor_memory(ctx['input_images'])
        del ctx['input_images']
