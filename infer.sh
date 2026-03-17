#!/bin/bash
PYTHONPATH=.
python inference/infer_ditto.py \
--lora_path models/ditto_global.safetensors \
--num_frames 73 \
--device_id 0 \
--input_video 测试数据.mp4 \
--output_video_dir results/3-17/ \
--prompt "Make it the LEGO style."