#!/usr/bin/env python3

from diffusers import AutoPipelineForText2Image, DiffusionPipeline
import torch
import json
import sys
import os

def load_weights(decoder_path, prior_path):
    pipe_prior = DiffusionPipeline.from_pretrained(prior_path, torch_dtype=torch.float16)
    prior_components = {"prior_" + k: v for k,v in pipe_prior.components.items()}
    pipe = AutoPipelineForText2Image.from_pretrained(decoder_path, **prior_components, torch_dtype=torch.float16)
    return pipe

def load_textprompts(jsonl_path):
    with open(jsonl_path, "r") as file:
        prompts = [json.loads(line) for line in file]
    return prompts

if __name__ == "__main__":
    decoder_path = sys.argv[1]
    prior_path = sys.argv[2]
    jsonl_path = sys.argv[3]
    target_folder = sys.argv[4]

    LIVECell_class_list = ['A172', 'BT474', 'BV2', 'Huh7', 'MCF7', 'SHSY5Y', 'SkBr3', 'SKOV3']
    cell_type_paths = dict()
    for cell_type in LIVECell_class_list:
        cell_type_paths[cell_type] = os.path.join(target_folder, cell_type)
        os.mkdir(cell_type_paths[cell_type])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = load_weights(decoder_path, prior_path)
    pipe.to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        pipe.unet = torch.nn.DataParallel(pipe.unet)

    prompts = load_textprompts(jsonl_path)

    # for prompt in prompts:
    #     file_name = prompt['file_name'].split('/')[1]
    #     cell_type = file_name.split('_')[0]
    #     out_path = os.path.join(cell_type_paths[cell_type], file_name)

    #     images = pipe(prompt=prompt['text']).images
    #     images[0].save(out_path)

    # Group prompts into batches
    batch_size = torch.cuda.device_count() * 2  # or another value that fits your GPU memory
    prompt_batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]

    for batch in prompt_batches:
        texts = [p['text'] for p in batch]
        file_names = [p['file_name'].split('/')[1] for p in batch]
        cell_types = [fn.split('_')[0] for fn in file_names]
        out_paths = [os.path.join(cell_type_paths[ct], fn) for ct, fn in zip(cell_types, file_names)]

        images = pipe(prompt=texts).images
        for img, out_path in zip(images, out_paths):
            img.save(out_path)
