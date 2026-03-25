import argparse
import torch
from PIL import Image
import os
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
import time


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def load_vace_reference_images_from_dir(dir_path: str, height: int, width: int):
    """
    Load all images in a folder as RGB PIL Images, sorted by filename.
    Resizes each to (width, height) to match pipeline inference size.
    """
    if not os.path.isdir(dir_path):
        return None
    paths = []
    for name in sorted(os.listdir(dir_path)):
        p = os.path.join(dir_path, name)
        if os.path.isfile(p) and os.path.splitext(name.lower())[1] in _IMAGE_EXTS:
            paths.append(p)
    if not paths:
        return None
    out = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        if im.size != (width, height):
            im = im.resize((width, height), Image.Resampling.LANCZOS)
        out.append(im)
    return out



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

    print(f"Loading input video: {args.input_video}")
    if not os.path.exists(args.input_video):
        print(f"Error: Input video file not found at {args.input_video}")
        return
        
    video = VideoData(args.input_video, height=args.height, width=args.width)
    
    num_frames = min(args.num_frames, len(video))
    if num_frames != args.num_frames:
        print(f"Warning: Requested number of frames ({args.num_frames}) exceeds total video frames ({len(video)}). Using {num_frames} frames instead.")
        
    video = [video[i] for i in range(num_frames)]
    
    reference_image = None

    vace_reference_image = None
    if args.vace_reference_dir:
        if not os.path.isdir(args.vace_reference_dir):
            print(f"Error: VACE reference directory not found: {args.vace_reference_dir}")
            return
        vace_reference_image = load_vace_reference_images_from_dir(
            args.vace_reference_dir, args.height, args.width
        )
        if not vace_reference_image:
            print(f"Error: No images ({', '.join(sorted(_IMAGE_EXTS))}) in {args.vace_reference_dir}")
            return
        print(f"Loaded {len(vace_reference_image)} VACE reference frame(s) from {args.vace_reference_dir}")



    start = time.perf_counter()

    video = pipe(
        prompt=args.prompt,
        negative_prompt="cgi, 3d, render, unreal engine, octane render, blender, digital art, plastic, wax, glossy, oily skin, airbrushed, photoshop, retouch, smooth skin, oversaturated, high contrast, vibrant, perfect, ideal, doll, mannequin, statue, shining, dreamy, fantasy, magical, cartoon, anime, illustration, sketch, painting, drawing, simplified, abstract, lowres, depth of field, bokeh, symmetry, centered, watermark, text, signature, blurry, low quality, artifacts, deformed, bad anatomy, artificial, fake, fake lighting, high saturation, grainless, manifold, jewelry, porcelain, synthetic, smooth texture, sharp edges, unreal lighting",
        vace_video=video,
        vace_reference_image=vace_reference_image,
        num_frames=num_frames,
        seed=args.seed,
        tiled=True,
    )

    end = time.perf_counter()
    print(f"Generation time: {end - start:.2f} seconds")
    duration = end - start
    output_dir = os.path.dirname(args.output_video_dir)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    save_video(video, f"{args.output_video_dir}/{args.input_video}_{num_frames}frame_{duration:.5f}.mp4", fps=args.fps, quality=args.quality)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InstructV2V Pipeline.")

    parser.add_argument("--input_video", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--output_video_dir", type=str, required=True, help="Path to save the output video file.")
    parser.add_argument("--lora_path", type=str, default=None, help="Optional path to a LoRA model file (.safetensors).")
    parser.add_argument("--device_id", type=int, default=0, help="The ID of the CUDA device to use (e.g., 0, 1, 2).")
    parser.add_argument("--prompt", type=str, required=True, help="The positive prompt describing the target style.")
    parser.add_argument("--height", type=int, default=480, help="The height to use for video processing.")
    parser.add_argument("--width", type=int, default=832, help="The width to use for video processing.")
    parser.add_argument("--num_frames", type=int, default=73, help="The number of video frames to process.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducible results.")

    parser.add_argument("--lora_alpha", type=float, default=1.0, help="The alpha (weight) value for the LoRA model.")
    parser.add_argument("--fps", type=int, default=20, help="Frames per second (FPS) for the output video.")
    parser.add_argument("--quality", type=int, default=5, help="Quality of the output video (CRF value, lower is better).")
    parser.add_argument(
        "--vace_reference_dir",
        type=str,
        default=None,
        help="Optional folder of images (png/jpg/...) as vace_reference_image, sorted by filename; resized to height x width.",
    )
    args = parser.parse_args()
    main(args)