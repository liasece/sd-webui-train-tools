import modules.scripts
from modules import sd_samplers
from modules.generation_parameters_copypaste import create_override_settings_dict
from modules.processing import StableDiffusionProcessing, Processed, StableDiffusionProcessingTxt2Img, \
    StableDiffusionProcessingImg2Img, process_images
from modules.shared import opts, cmd_opts
import modules.shared as shared
import modules.processing as processing
import modules.images as images
from modules.ui import plaintext_to_html
from liasece_sd_webui_train_tools.util import *
import time
import os
import sys
import traceback

import liasece_sd_webui_train_tools.tools.xyz_grid as xyz_grid

def preview_checkpoint(save_file_path: str, checkpoint_name: str, checkpoint_path: str, 
        # preview view config
        preview_include_sub_img: bool,
        # txt2txt
        preview_txt2img_prompt: str, # like "apple"
        preview_txt2img_negative_prompt: str, # like "apple"
        preview_sampling_method: list[str], # like `"Euler a", "ms"`
        preview_sampling_steps: str, # like 20,24,28
        preview_width: int, # like 512
        preview_height: int, # like 512
        preview_batch_count: int, # like 1
        preview_batch_size: int, # like 1
        preview_cfg_scale: str, # like 8,9,10,11
        preview_seed: str, # like -1,-1
        preview_lora_multiplier: str, # like 0.6,0.7,0.8,0.9
    ):
    preview_lora_multiplier_list = preview_lora_multiplier.split(",")
    assert len(preview_lora_multiplier_list) > 0, "preview_lora_multiplier must be a list of at least one element"
    lora_dir = shared.cmd_opts.lora_dir
    shared.cmd_opts.lora_dir = os.path.abspath(os.path.dirname(checkpoint_path))
    default_lora_multiplier = preview_lora_multiplier_list[0]
    def lora_prompt(name: str, multiplier: str) -> str:
        return f"<lora:{name}:{multiplier}>"
    def lora_prompt_list(name: str, multiplier_list: list[str]) -> list[str]:
        return [lora_prompt(name, multiplier) for multiplier in multiplier_list]
    preview_txt2img_prompt = lora_prompt(checkpoint_name, default_lora_multiplier)+", " + preview_txt2img_prompt

    outpath_samples = save_file_path if preview_include_sub_img else None
    p = StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        outpath_samples=outpath_samples or opts.outdir_samples or opts.outdir_txt2img_samples,
        outpath_grids=save_file_path or opts.outdir_grids or opts.outdir_txt2img_grids,
        prompt=preview_txt2img_prompt,
        styles="",
        negative_prompt=preview_txt2img_negative_prompt,
        seed=int(preview_seed.split(",")[0].strip()),
        subseed=0,
        subseed_strength=0,
        seed_resize_from_h=0,
        seed_resize_from_w=0,
        seed_enable_extras=0,
        sampler_name=preview_sampling_method[0],
        batch_size=preview_batch_size,
        n_iter=preview_batch_count,
        steps=int(preview_sampling_steps.split(",")[0].strip()),
        cfg_scale=float(preview_cfg_scale.split(",")[0].strip()),
        width=preview_width,
        height=preview_height,
        restore_faces=False,
        tiling=False,
        enable_hr=False,
        denoising_strength=None,
        hr_scale=0,
        hr_upscaler="",
        hr_second_pass_steps=0,
        hr_resize_x=0,
        hr_resize_y=0,
        override_settings={},
    )

    # p.scripts = modules.scripts.scripts_txt2img

    # last_arg_index = 1
    # for script in p.scripts.scripts:
    #     if last_arg_index < script.args_to:
    #         last_arg_index = script.args_to
    # p.script_args = [None]*last_arg_index

    runner = xyz_grid.Script()
    runner.current_axis_options = [x for x in xyz_grid.axis_options if type(x) == xyz_grid.AxisOption or not x.is_img2img]
    def find_index_in_current_axis_options(label):
        for i, x in enumerate(runner.current_axis_options):
            if x.label == label:
                return i
        return -1
    args = []
    if len(preview_sampling_method) > 1:
        args+=[find_index_in_current_axis_options("Sampler"), ",".join(preview_sampling_method)]
    if len(preview_sampling_steps.split(",")) > 1:
        args+=[find_index_in_current_axis_options("Steps"), preview_sampling_steps]
    if len(preview_cfg_scale.split(",")) > 1:
        args+=[find_index_in_current_axis_options("CFG Scale"), preview_cfg_scale]
    if len(preview_seed.split(",")) > 1:
        args+=[find_index_in_current_axis_options("Seed"), preview_seed]
    if len(preview_lora_multiplier_list) > 1:
        args+=[find_index_in_current_axis_options("Prompt S/R"), ",".join(lora_prompt_list(checkpoint_name, preview_lora_multiplier_list))]
    assert len(args) <= 2*3, "Too many xyz_grid parameters to preview"
    while len(args) < 6:
        args+=[0, ""]
    args+=[
        True, # draw_legend
        False, # include_lone_images
        False, # include_sub_grids
        False, # no_fixed_seeds
        0, # margin_size
    ]
    printD(f"txt2img: {preview_txt2img_prompt}", args)
    try:
        processed = runner.run(p, *args)
        if processed is None:
            processed = process_images(p)
        p.close()
        shared.total_tqdm.clear()
        generation_info_js = processed.js()
        printD(generation_info_js)
        if opts.do_not_show_images:
            processed.images = []
    except Exception as e:
        printD(f"Error while processing {p.prompt}", e)
        print(traceback.format_exc(), file=sys.stderr)

    shared.cmd_opts.lora_dir = lora_dir
