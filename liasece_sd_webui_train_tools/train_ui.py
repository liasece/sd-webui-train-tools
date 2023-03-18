# -*- coding: UTF-8 -*-

import gradio as gr
import os
from modules import shared
from modules import sd_models
from liasece_sd_webui_train_tools.util import *
from liasece_sd_webui_train_tools import train
from liasece_sd_webui_train_tools import ArgsList

from liasece_sd_webui_train_tools.project import *
from liasece_sd_webui_train_tools.config_file import *
# from liasece_sd_webui_train_tools.project_version_ui import *
# from liasece_sd_webui_train_tools.ui import *
from liasece_sd_webui_train_tools.checkpoint_preview_ui import *
# from liasece_sd_webui_train_tools.dateset_ui import *
# from liasece_sd_webui_train_tools.train_ui import *

def on_train_begin_click(id: str, project: str, version: str,
        # train config
        train_base_model: str, 
        train_batch_size: int, 
        train_num_epochs: int, 
        train_save_every_n_epochs: int,
        train_finish_generate_all_checkpoint_preview: bool,
        train_optimizer_type: str,
        train_learning_rate: float,
        train_net_dim: int,
        train_alpha: int,
        train_clip_skip: int,
        train_mixed_precision: str,
        train_xformers: bool,
        train_base_on_sd_v2: bool,
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
    save_train_config(project, version, {
        # train config
        "train_base_model": train_base_model,
        "train_batch_size": train_batch_size,
        "train_num_epochs": train_num_epochs,
        "train_save_every_n_epochs": train_save_every_n_epochs,
        "train_finish_generate_all_checkpoint_preview": train_finish_generate_all_checkpoint_preview,
        "train_optimizer_type": train_optimizer_type,
        "train_learning_rate": train_learning_rate,
        "train_net_dim": train_net_dim,
        "train_alpha": train_alpha,
        "train_clip_skip": train_clip_skip,
        "train_mixed_precision": train_mixed_precision,
        "train_xformers": train_xformers,
        "train_base_on_sd_v2": train_base_on_sd_v2,
    })
    save_preview_config(project, version, {
        # preview view config
        "preview_include_sub_img": preview_include_sub_img,
        # txt2txt
        "preview_txt2img_prompt": preview_txt2img_prompt, # like "apple"
        "preview_txt2img_negative_prompt": preview_txt2img_negative_prompt, # like "apple"
        "preview_sampling_method": preview_sampling_method, # like `"Euler a", "ms"`
        "preview_sampling_steps": preview_sampling_steps, # like 20,24,28
        "preview_width": preview_width, # like 512
        "preview_height": preview_height, # like 512
        "preview_batch_count": preview_batch_count, # like 1
        "preview_batch_size": preview_batch_size, # like 1
        "preview_cfg_scale": preview_cfg_scale, # like 8,9,10,11
        "preview_seed": preview_seed, # like -1,-1
        "preview_lora_multiplier": preview_lora_multiplier, # like 0.6,0.7,0.8,0.9
    })
    train_base_model_path = ""
    for x in sd_models.checkpoints_list.values():
        if x.title == train_base_model:
            train_base_model_path = os.path.join(sd_models.model_path, x.name)
            break
    processed_path = get_project_version_dataset_processed_path(project, version)
    os.makedirs(processed_path, exist_ok=True)
    project_version_checkpoint_path = get_project_version_checkpoint_path(project, version)
    os.makedirs(project_version_checkpoint_path, exist_ok=True)

    cfg = ArgsList.ArgStore()
    cfg.img_folder = os.path.abspath(processed_path)
    cfg.output_folder = os.path.abspath(project_version_checkpoint_path)
    cfg.change_output_name = project+r"-"+version
    cfg.batch_size = int(train_batch_size)
    cfg.num_epochs = int(train_num_epochs)
    cfg.save_every_n_epochs = int(train_save_every_n_epochs)
    cfg.base_model = train_base_model_path
    cfg.optimizer_type = train_optimizer_type
    cfg.learning_rate = float(train_learning_rate)
    cfg.net_dim = int(train_net_dim)
    cfg.alpha = int(train_alpha)
    cfg.clip_skip = int(train_clip_skip)
    cfg.mixed_precision = train_mixed_precision
    cfg.xformers = train_xformers
    cfg.v2 = train_base_on_sd_v2
    printD("on_train_begin_click", cfg.__dict__)
    train.train(cfg)
    if train_finish_generate_all_checkpoint_preview:
        return [None]+on_ui_preview_generate_all_preview_btn_click(project, version,
            preview_include_sub_img,
            preview_txt2img_prompt,
            preview_txt2img_negative_prompt,
            preview_sampling_method,
            preview_sampling_steps,
            preview_width,
            preview_height,
            preview_batch_count,
            preview_batch_size,
            preview_cfg_scale,
            preview_seed,
            preview_lora_multiplier,
        )
    return [None]+gr_update_checkpoint_list(project, version)+["done"]

def ui_refresh_train_base_model():
    shared.refresh_checkpoints()
    tiles = shared.list_checkpoint_tiles()
    return [
        gr.Dropdown.update(visible=True, choices=tiles,value = tiles[0] if len(tiles) > 0 else ""),
    ]
