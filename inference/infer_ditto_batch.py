import argparse
import json
import torch
from PIL import Image
import os
import time
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig


def process_video(pipe, args, first_prompt, rest_prompt, output_dir):
    print(f"Loading input video: {args.input_video}")
    if not os.path.exists(args.input_video):
        print(f"Error: Input video file not found at {args.input_video}")
        return

    original = VideoData(args.input_video, height=args.height, width=args.width)
    total_frames = len(original)
    print(f"Total frames in video: {total_frames}")

    segment_size = args.segment_size
    overlap = args.overlap_frames
    stride = segment_size - overlap

    if segment_size <= overlap:
        print(f"Error: segment_size ({segment_size}) must be larger than overlap ({overlap})")
        return

    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "checkpoint.json")

    # ── Resume from checkpoint if available ──
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path) as f:
                ckpt = json.load(f)
            resume_idx = ckpt["segment_idx"]
            curr_start = ckpt["curr_start"]

            stitched_path = os.path.join(output_dir, "stitched.mp4")
            if os.path.exists(stitched_path):
                stitched = VideoData(stitched_path, height=args.height, width=args.width)
                all_output_frames = [stitched[i] for i in range(len(stitched))]
            else:
                all_output_frames = []

            last_seg_path = os.path.join(output_dir, f"seg_{resume_idx:04d}.mp4")
            if os.path.exists(last_seg_path) and resume_idx > 0:
                last_seg = VideoData(last_seg_path, height=args.height, width=args.width)
                prev_overlap_frames = [last_seg[i] for i in range(max(0, len(last_seg) - args.overlap_frames), len(last_seg))]
            else:
                prev_overlap_frames = None

            segment_idx = resume_idx
            print(f"\n[Resume] Skipping segments 1–{resume_idx}, starting at segment {resume_idx + 1}")
            print(f"[Resume] Loaded {len(all_output_frames)} accumulated frames, "
                  f"overlap={len(prev_overlap_frames) if prev_overlap_frames else 0}")
        except Exception as e:
            print(f"[Resume] Checkpoint corrupt ({e}), starting fresh.")
            all_output_frames = []
            prev_overlap_frames = None
            curr_start = 0
            segment_idx = 0
    else:
        all_output_frames = []
        prev_overlap_frames = None
        curr_start = 0
        segment_idx = 0

    total_start = time.perf_counter()

    while curr_start < total_frames:
        segment_idx += 1
        remaining = total_frames - curr_start

        if segment_idx == 1:
            n_frames = min(segment_size, remaining)
            input_frames = [original[i] for i in range(n_frames)]
            n_prefix = 0
        else:
            n_new = min(stride, remaining)
            input_frames = list(prev_overlap_frames) + [original[curr_start + i] for i in range(n_new)]
            n_prefix = len(prev_overlap_frames)

        actual_segment_size = len(input_frames)
        if n_prefix > actual_segment_size:
            n_prefix = actual_segment_size

        print(f"\n{'='*60}")
        print(f"Segment {segment_idx}: {actual_segment_size} frames, prefix={n_prefix}, "
              f"original_range=[{curr_start}, {curr_start + actual_segment_size - n_prefix})")
        print(f"{'='*60}")

        vace_video_mask = []
        for i in range(actual_segment_size):
            pixel_value = 0 if i < n_prefix else 255
            vace_video_mask.append(
                Image.new("RGB", (args.width, args.height), (pixel_value, pixel_value, pixel_value))
            )

        seg_start = time.perf_counter()
        output = pipe(
            prompt=first_prompt if segment_idx == 1 else rest_prompt,
            negative_prompt=args.negative_prompt or "",
            cfg_scale=args.cfg_scale,
            vace_video=input_frames,
            vace_video_mask=vace_video_mask,
            num_frames=actual_segment_size,
            seed=args.seed if not args.increment_seed else args.seed + segment_idx,
            tiled=True,
        )
        seg_duration = time.perf_counter() - seg_start
        print(f"Segment {segment_idx} generated in {seg_duration:.2f}s")

        new_frames = list(output[n_prefix:])
        all_output_frames.extend(new_frames)
        print(f"Collected {len(new_frames)} new frames (total output: {len(all_output_frames)})")

        # Save this segment's full output (with overlap)
        seg_path = os.path.join(output_dir, f"seg_{segment_idx:04d}.mp4")
        save_video(output, seg_path, fps=args.fps, quality=args.quality)
        print(f"Saved segment: {seg_path}")

        # Save accumulated stitched result (survives interruption)
        stitched_path = os.path.join(output_dir, "stitched.mp4")
        save_video(all_output_frames, stitched_path, fps=args.fps, quality=args.quality)

        if remaining > (segment_size if segment_idx == 1 else stride):
            prev_overlap_frames = list(output[-overlap:])
        else:
            prev_overlap_frames = None

        if segment_idx == 1:
            curr_start += n_frames
        else:
            curr_start += min(stride, remaining)

        with open(checkpoint_path, 'w') as f:
            json.dump({"segment_idx": segment_idx, "curr_start": curr_start}, f)

    total_duration = time.perf_counter() - total_start
    print(f"\n{'='*60}")
    print(f"All segments done. Total time: {total_duration:.2f}s")
    print(f"Total output frames: {len(all_output_frames)}")
    print(f"Output directory: {output_dir}")

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)


def main(args):
    device = f"cuda:{args.device_id}"

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
        ],
    )
    if args.lora_path:
        print(f"Loading Ditto LoRA model: {args.lora_path} (alpha={args.lora_alpha})")
        if not os.path.exists(args.lora_path):
            print(f"Error: LoRA file not found at {args.lora_path}")
            return
        pipe.load_lora(pipe.vace, args.lora_path, alpha=args.lora_alpha)

    pipe.enable_vram_management()

    # ── Build experiment list ──
    if args.first_prompt is not None and args.rest_prompt is not None:
        if len(args.first_prompt) != len(args.rest_prompt):
            print(f"Error: --first_prompt ({len(args.first_prompt)} entries) and "
                  f"--rest_prompt ({len(args.rest_prompt)} entries) count mismatch.")
            return
        experiments = list(zip(args.first_prompt, args.rest_prompt))
    elif args.prompt is not None:
        experiments = [(args.prompt, args.prompt)]
    else:
        print("Error: Must provide --prompt, or both --first_prompt and --rest_prompt.")
        return

    num_experiments = len(experiments)
    print(f"Running {num_experiments} experiment(s)")

    for exp_idx, (first_prompt, rest_prompt) in enumerate(experiments):
        if num_experiments > 1:
            print(f"\n{'#'*60}")
            print(f"Experiment {exp_idx + 1}/{num_experiments}")
            print(f"  First-segment prompt: {first_prompt}")
            print(f"  Rest-segment prompt:  {rest_prompt}")
            print(f"{'#'*60}")

        if num_experiments == 1:
            exp_output_dir = args.output_dir
        else:
            exp_output_dir = os.path.join(args.output_dir, f"exp_{exp_idx:03d}")

        process_video(pipe, args, first_prompt, rest_prompt, exp_output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ditto batch processing — long video with sliding window.")

    # I/O
    parser.add_argument("--input_video", type=str, required=True, help="Path to the full input video file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output videos.")
    parser.add_argument("--lora_path", type=str, default=None, help="Optional path to a LoRA model file (.safetensors).")

    # Model
    parser.add_argument("--device_id", type=int, default=0, help="CUDA device ID.")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt used for all segments (shortcut for single experiment).")
    parser.add_argument("--first_prompt", type=str, action="append", default=None, help="Prompt for the first segment. Repeat for multiple experiments.")
    parser.add_argument("--rest_prompt", type=str, action="append", default=None, help="Prompt for segments 2+. Repeat for multiple experiments.")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative prompt.")
    parser.add_argument("--cfg_scale", type=float, default=2.6, help="CFG scale.")
    parser.add_argument("--lora_alpha", type=float, default=1.0, help="LoRA alpha.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--increment_seed", action="store_true", help="Increment seed per segment (default: use same seed).")

    # Video
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--fps", type=int, default=20, help="Output video FPS.")
    parser.add_argument("--quality", type=int, default=5, help="Output video quality (CRF).")

    # Segmentation
    parser.add_argument("--segment_size", type=int, default=73, help="Frames per segment.")
    parser.add_argument("--overlap_frames", type=int, default=21, help="Overlap frames between consecutive segments.")

    args = parser.parse_args()
    main(args)
