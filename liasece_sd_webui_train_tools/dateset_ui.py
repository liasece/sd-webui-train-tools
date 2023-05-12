# -*- coding: UTF-8 -*-

import gradio as gr
import os
import shutil
from PIL import Image
import tempfile
import modules
import sys
import traceback
from liasece_sd_webui_train_tools.util import *
from liasece_sd_webui_train_tools import no_alpha_0_picture

from liasece_sd_webui_train_tools.project import *
from liasece_sd_webui_train_tools.config_file import *
import liasece_sd_webui_train_tools.tools.preprocess as preprocess
# from liasece_sd_webui_train_tools.project_version_ui import *
# from liasece_sd_webui_train_tools.ui import *
# from liasece_sd_webui_train_tools.checkpoint_preview_ui import *
# from liasece_sd_webui_train_tools.dateset_ui import *
# from liasece_sd_webui_train_tools.train_ui import *

def on_ui_update_dataset_click(id: str, project: str, version: str, input_train_data_set_files: list[tempfile._TemporaryFileWrapper], train_num_repetitions: int, 
        process_width,
        process_height,
        preprocess_txt_action,
        process_flip,
        process_split,
        process_caption,
        process_caption_deepbooru,
        process_split_threshold,
        process_overlap_ratio,
        process_focal_crop,
        process_focal_crop_face_weight,
        process_focal_crop_entropy_weight,
        process_focal_crop_edges_weight,
        process_focal_crop_debug,
        process_multicrop,
        process_multicrop_mindim,
        process_multicrop_maxdim,
        process_multicrop_minarea,
        process_multicrop_maxarea,
        process_multicrop_objective,
        process_multicrop_threshold,
    ) -> list[Image.Image]:
    save_dataset_config(project, version, {
        "train_num_repetitions": train_num_repetitions,
        "process_width": process_width,
        "process_height": process_height,
        "preprocess_txt_action": preprocess_txt_action,
        "process_flip": process_flip,
        "process_split": process_split,
        "process_caption": process_caption,
        "process_caption_deepbooru": process_caption_deepbooru,
        "process_split_threshold": process_split_threshold,
        "process_overlap_ratio": process_overlap_ratio,
        "process_focal_crop": process_focal_crop,
        "process_focal_crop_face_weight": process_focal_crop_face_weight,
        "process_focal_crop_entropy_weight": process_focal_crop_entropy_weight,
        "process_focal_crop_edges_weight": process_focal_crop_edges_weight,
        "process_focal_crop_debug": process_focal_crop_debug,
        "process_multicrop": process_multicrop,
        "process_multicrop_mindim": process_multicrop_mindim,
        "process_multicrop_maxdim": process_multicrop_maxdim,
        "process_multicrop_minarea": process_multicrop_minarea,
        "process_multicrop_maxarea": process_multicrop_maxarea,
        "process_multicrop_objective": process_multicrop_objective,
        "process_multicrop_threshold": process_multicrop_threshold,
    })
    train_num_repetitions = int(train_num_repetitions)
    if input_train_data_set_files is None or len(input_train_data_set_files) == 0:
        return
    
    # save to project version path
    origin_data_path = get_project_version_dataset_origin_path(project, version)
    origin_preload_data_path = os.path.join(origin_data_path, "preload")
    processed_path = get_project_version_dataset_processed_path(project, version)
    printD("on_ui_update_dataset_click save_path:", origin_data_path, processed_path)
    if origin_data_path == "" or processed_path == "":
        return

    # remove old dataset
    if os.path.isdir(origin_data_path):
        shutil.rmtree(origin_data_path)
    if os.path.isdir(processed_path):
        shutil.rmtree(processed_path)

    # save new dataset
    os.makedirs(origin_data_path, exist_ok=True)
    os.makedirs(processed_path, exist_ok=True)
    for f in input_train_data_set_files:
        file_path = os.path.join(origin_data_path, os.path.basename(f.name))
        printD("on_ui_update_dataset_click file:", file_path)
        # copy to dataset path
        shutil.copyfile(f.name, file_path)
    if train_num_repetitions < 0:
        if len(input_train_data_set_files) > 0:
            train_num_repetitions = int(512 / len(input_train_data_set_files))
        if train_num_repetitions <= 4:
            train_num_repetitions = 4
    processed_output_path = os.path.join(processed_path, str(train_num_repetitions)+"_"+project)
    os.makedirs(processed_output_path, exist_ok=True)
    os.makedirs(origin_preload_data_path, exist_ok=True)

    try:
        no_alpha_0_picture.rangeAllImage(origin_data_path, origin_preload_data_path)
        preprocess.preprocess(None, origin_preload_data_path, processed_output_path, 
            process_width = process_width,
            process_height = process_height,
            preprocess_txt_action = preprocess_txt_action,
            process_keep_original_size = False,
            process_flip = process_flip,
            process_split = process_split,
            process_caption = process_caption,
            process_caption_deepbooru = process_caption_deepbooru,
            split_threshold = process_split_threshold,
            overlap_ratio = process_overlap_ratio,
            process_focal_crop = process_focal_crop,
            process_focal_crop_face_weight = process_focal_crop_face_weight,
            process_focal_crop_entropy_weight = process_focal_crop_entropy_weight,
            process_focal_crop_edges_weight = process_focal_crop_edges_weight,
            process_focal_crop_debug = process_focal_crop_debug,
            process_multicrop = process_multicrop,
            process_multicrop_mindim = process_multicrop_mindim,
            process_multicrop_maxdim = process_multicrop_maxdim,
            process_multicrop_minarea = process_multicrop_minarea,
            process_multicrop_maxarea = process_multicrop_maxarea,
            process_multicrop_objective = process_multicrop_objective,
            process_multicrop_threshold = process_multicrop_threshold,
        )
    except Exception as e:
        printD("dataset update error", e)
        print(traceback.format_exc(), file=sys.stderr)

    return get_project_version_dataset_box_update(project, version)+[""]

def get_project_version_dataset_box_update(project: str, version: str):
    processed_output_path = get_project_version_dataset_processed_path(project, version)
    if processed_output_path == "":
        return [ gr.Row.update(visible=False)]*3+[None]*2
    dataset_images = readImagePaths(processed_output_path, 1)
    label_head = f"Dataset: {len(dataset_images)} images"
    sub_dir_list = readPathSubDirPathList(processed_output_path)
    label = "\n".join(sub_dir_list)
    return [
        gr.Row.update(visible=version!=""), # gr_project_version_dateset_row
        gr.Row.update(visible=version!=""), # train_row
        gr.Box.update(visible=version!=""), # preview_box
        gr.Gallery.update(value=dataset_images), # gr_project_version_dataset_gallery
        gr.Textbox.update(value=f"{label_head}\n{label}", lines=1+len(sub_dir_list)), # gr_project_version_dataset_label
    ]
