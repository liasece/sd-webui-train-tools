# -*- coding: UTF-8 -*-

import gradio as gr
import os
import shutil
import random
import time
from liasece_sd_webui_train_tools.util import *
from liasece_sd_webui_train_tools import preview
import modules.images as images

from liasece_sd_webui_train_tools.project import *
from liasece_sd_webui_train_tools.config_file import *
# from liasece_sd_webui_train_tools.project_version_ui import *
# from liasece_sd_webui_train_tools.ui import *
# from liasece_sd_webui_train_tools.checkpoint_preview_ui import *
# from liasece_sd_webui_train_tools.dateset_ui import *
# from liasece_sd_webui_train_tools.train_ui import *

max_list_checkpoint = 10

# list checkpoint, [(name, path)]
def list_checkpoint(project: str, version: str, train_name: str) -> list[(str, str)]:
    checkpoint_path = get_project_version_trains_checkpoint_path(project, version, train_name)
    if checkpoint_path == "":
        return []
    return readCheckpoints(checkpoint_path)

def get_checkpoint_preview_images_path(project: str, version: str, train_name: str, checkpoint_name: str):
    checkpoint_path = get_project_version_trains_checkpoint_path(project, version, train_name)
    if checkpoint_path == "":
        return ""
    if checkpoint_name == "":
        return ""
    return os.path.join(checkpoint_path, checkpoint_name, "images")

def get_checkpoint_preview_images(project: str, version: str, train_name: str, checkpoint_name: str):
    checkpoint_preview_images_path = get_checkpoint_preview_images_path(project, version, train_name, checkpoint_name)
    if checkpoint_preview_images_path == "":
        return []
    # printD(f"in get_checkpoint_preview_images: project: {project}-{version}-{train_name}-{checkpoint_name}")
    xyz_grid = readImagePaths(checkpoint_preview_images_path, level=3, include_pre_level=True, endswith=".png", startswith="xyz_grid-")
    sub_grid = readImagePaths(checkpoint_preview_images_path, level=3, include_pre_level=True, endswith=".png", startswith="grid-")
    all_img = readImagePaths(checkpoint_preview_images_path, level=3, include_pre_level=True, endswith=".png")
    printD(f"in get_checkpoint_preview_images: project: {project} {version} {train_name} {checkpoint_name}: xyz_grid {len(xyz_grid)}, sub_grid {len(sub_grid)}, all_img {len(all_img)}")
    if len(sub_grid) + len(xyz_grid) != len(all_img):
        return all_img
    return xyz_grid

def get_checkpoint_preview_images_update(project: str, version: str, train_name: str, checkpoint_name: str):
    images = get_checkpoint_preview_images(project, version, train_name, checkpoint_name)
    return [
        gr.Gallery.update(value=images, visible=len(images)>1), 
        gr.Image.update(value=images[0] if len(images)>0 else None, visible=len(images)==1),
    ]

# list checkpoint, [(name, path)]
def gr_update_trains_area_list(project: str, version: str, train_name: str):
    project_version_train_list = get_project_version_trains_list(project, version)
    to_train_name = train_name
    if to_train_name == "" or to_train_name == None:
        to_train_name = project_version_train_list[0] if len(project_version_train_list) > 0 else ""
    return [gr.Dropdown.update(value=to_train_name if to_train_name != "" else None, choices=project_version_train_list)]+gr_update_checkpoint_list(project, version, to_train_name)

# list checkpoint, [(name, path)]
def gr_update_checkpoint_list(project: str, version: str, train_name: str):
    train_checkpoint_row_list=[]
    train_checkpoint_info_name_list = []
    train_checkpoint_info_path_list = []
    train_checkpoint_txt2txt_preview_gallery_list = []
    train_checkpoint_txt2txt_preview_image_list = []
    checkpoint_list = list_checkpoint(project, version, train_name)
    # printD("in gr_update_checkpoint_list: checkpoint_list", checkpoint_list)
    for i in range(0, max_list_checkpoint):
        visible = i<len(checkpoint_list)
        checkpoint_name = checkpoint_list[i][0] if visible else ""
        checkpoint_path = checkpoint_list[i][1] if visible else ""
        train_checkpoint_info_name_list.append(gr.Textbox.update(checkpoint_name))
        train_checkpoint_info_path_list.append(gr.Textbox.update(checkpoint_path))
        update = get_checkpoint_preview_images_update(project, version, train_name, checkpoint_name)
        train_checkpoint_txt2txt_preview_gallery_list.append(update[0])
        train_checkpoint_txt2txt_preview_image_list.append(update[1])
        train_checkpoint_row_list.append(gr.Row.update(visible=visible))
    return train_checkpoint_row_list+train_checkpoint_info_name_list+train_checkpoint_info_path_list+train_checkpoint_txt2txt_preview_gallery_list+train_checkpoint_txt2txt_preview_image_list

def on_ui_preview_1_checkpoint_click(project: str, version: str, train_name: str, checkpoint_name: str, checkpoint_path: str, before_delete_preview_images: bool,
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
    save_preview_config(project, version, {
        # preview view config
        "preview_include_sub_img":preview_include_sub_img,
        # txt2txt
        "preview_txt2img_prompt":preview_txt2img_prompt, # like "apple"
        "preview_txt2img_negative_prompt":preview_txt2img_negative_prompt, # like "apple"
        "preview_sampling_method":preview_sampling_method, # like `"Euler a", "ms"`
        "preview_sampling_steps":preview_sampling_steps, # like 20,24,28
        "preview_width":preview_width, # like 512
        "preview_height":preview_height, # like 512
        "preview_batch_count":preview_batch_count, # like 1
        "preview_batch_size":preview_batch_size, # like 1
        "preview_cfg_scale":preview_cfg_scale, # like 8,9,10,11
        "preview_seed":preview_seed, # like -1,-1
        "preview_lora_multiplier":preview_lora_multiplier, # like 0.6,0.7,0.8,0.9
    })
    save_image_path = get_checkpoint_preview_images_path(project, version, train_name, checkpoint_name)
    if before_delete_preview_images:
        if os.path.isdir(save_image_path):
            shutil.rmtree(save_image_path)
    preview.preview_checkpoint(save_image_path, checkpoint_name, checkpoint_path, 
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
    wait_image_save_finished()
    return get_checkpoint_preview_images_update(project, version, train_name, checkpoint_name)+[
        "done",
    ]

def on_ui_delete_all_preview_images_click(project: str, version: str, selected_train_name: str, checkpoint_name: str):
    trains_list = get_project_version_trains_list(project, version)
    for train_name in trains_list:
        for checkpoint_name, checkpoint_path in list_checkpoint(project, version, train_name):
            path = get_checkpoint_preview_images_path(project, version, train_name, checkpoint_name)
            if os.path.isdir(path):
                shutil.rmtree(path)
    return gr_update_trains_area_list(project, version, selected_train_name)

def on_ui_preview_generate_all_preview_btn_click(id: str, project: str, version: str, selected_train_name: str,
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
    save_preview_config(project, version, {
        # preview view config
        "preview_include_sub_img":preview_include_sub_img,
        # txt2txt
        "preview_txt2img_prompt":preview_txt2img_prompt, # like "apple"
        "preview_txt2img_negative_prompt":preview_txt2img_negative_prompt, # like "apple"
        "preview_sampling_method":preview_sampling_method, # like `"Euler a", "ms"`
        "preview_sampling_steps":preview_sampling_steps, # like 20,24,28
        "preview_width":preview_width, # like 512
        "preview_height":preview_height, # like 512
        "preview_batch_count":preview_batch_count, # like 1
        "preview_batch_size":preview_batch_size, # like 1
        "preview_cfg_scale":preview_cfg_scale, # like 8,9,10,11
        "preview_seed":preview_seed, # like -1,-1
        "preview_lora_multiplier":preview_lora_multiplier, # like 0.6,0.7,0.8,0.9
    })
    # turn -1 seed to random
    preview_seed_list = preview_seed.split(",")
    for i in range(0, len(preview_seed_list)):
        if int(preview_seed_list[i].strip())==-1:
            preview_seed_list[i] = str(random.randint(0, 2400000000))
    preview_seed = ",".join(preview_seed_list)
    trains_list = get_project_version_trains_list(project, version)
    for train_name in trains_list:
        checkpoint_list = list_checkpoint(project, version, train_name)
        i=0
        for checkpoint_name, checkpoint_path in checkpoint_list:
            i+=1
            printD(f"preview checkpoint: {checkpoint_name}({checkpoint_path}) {i}/{len(checkpoint_list)}")
            save_image_path = get_checkpoint_preview_images_path(project, version, train_name, checkpoint_name)
            # delete old preview
            if os.path.isdir(save_image_path):
                shutil.rmtree(save_image_path)
            preview.preview_checkpoint(save_image_path, checkpoint_name, checkpoint_path, 
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
    wait_image_save_finished()
    return gr_update_trains_area_list(project, version, selected_train_name)+[""]

def wait_image_save_finished():
    if hasattr(images,"is_async_save_image_finished"):
        printD(f"waiting for async save image finished...")
        # wait for async save to finish
        while not images.is_async_save_image_finished():
            time.sleep(0.5)

def on_ui_change_project_version_trains_click(project: str, version: str, train_name: str):
    return gr_update_trains_area_list(project, version, train_name)

def on_ui_refresh_project_version_trains_click(project: str, version: str, train_name: str):
    return gr_update_trains_area_list(project, version, train_name)
