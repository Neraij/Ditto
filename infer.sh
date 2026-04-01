#!/bin/bash
export PYTHONPATH=.
python inference/infer_ditto.py \
--lora_path models/ditto_global.safetensors \
--num_frames 73 \
--device_id 0 \
--input_video 测试数据_0001.mp4 \
--output_video_dir results/3-27/ \
--prompt "A hyper-realistic cinematic shot of a heavy rainy day, maintaining the original scene structure. The road surface is wet asphalt with realistic puddles and accurate water reflections. Detailed falling rain streaks and a dense atmospheric rain veil fill the air. Cinematic lighting, overcast sky, hyper-detailed textures, 8k resolution, photorealistic.
" \
--vace_reference_dir reference_folder/