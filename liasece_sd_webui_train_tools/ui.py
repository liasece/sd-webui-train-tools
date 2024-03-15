# -*- coding: UTF-8 -*-

import modules.scripts as scripts
import gradio as gr
from modules import shared
from modules import ui
from modules import sd_samplers
from liasece_sd_webui_train_tools.util import *
from modules.call_queue import wrap_gradio_gpu_call, wrap_queued_call, wrap_gradio_call

from liasece_sd_webui_train_tools.project import *
from liasece_sd_webui_train_tools.project_version_ui import *
from liasece_sd_webui_train_tools.ui import *
from liasece_sd_webui_train_tools.checkpoint_preview_ui import *
from liasece_sd_webui_train_tools.dateset_ui import *
from liasece_sd_webui_train_tools.train_ui import *

def new_ui():
    # ====UI====
    # prepare for train data set
    with gr.Blocks(css="button {background-color: #2171f1}") as train_tools:
        # check point list
        train_checkpoint_row_list= []
        train_checkpoint_info_name_list= []
        train_checkpoint_info_path_list= []
        train_checkpoint_txt2txt_preview_gallery_list= []
        train_checkpoint_txt2txt_preview_single_image_list= []
        train_checkpoint_txt2txt_preview_btn_list= []
        train_checkpoint_txt2txt_preview_delete_before_generate_list= []
        train_checkpoint_txt2txt_preview_btn_log_list= []

        dummy_component = gr.Label(visible=False)
        with gr.Row():
            # UI: Project Dropdown
            with gr.Box():  
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            # interactive: must true, this used in create_project_btn click event output
                            gr_project_dropdown = gr.Dropdown([], label=f"Project", value=None, interactive=True)
                    with gr.Column():
                        with gr.Row():
                            project_refresh_button = ui.ToolButton(value=ui.refresh_symbol, elem_id="project_refresh_button")
                            create_project_btn = gr.Button(value="Create Project", variant="primary")
        with gr.Row(visible=False) as gr_project_version_row:
            # UI: Project version Dropdown
            with gr.Box():  
                with gr.Row():
                    # UI: Project Version Dropdown
                    with gr.Column():
                        with gr.Row():
                            gr_version_dropdown = gr.Dropdown([], label=f"Version", value=None, interactive=True)
                    with gr.Column():
                        with gr.Row():
                            project_version_refresh_button = ui.ToolButton(value=ui.refresh_symbol, elem_id="refresh_gr_version_dropdown")
                            create_project_version_btn = gr.Button(value="Create Version", variant="primary")
        with gr.Row(visible=False) as gr_project_version_dateset_row:
            # UI: Project Version Dataset Dropdown
            with gr.Column():
                # UI: current dataset images
                with gr.Box():
                    gr.Markdown(f"### Current Dataset to be trained")
                    with gr.Row():
                        gr_project_version_dataset_gallery = gr.Gallery(value=None, label='Output', show_label=False, elem_id=f"gr_project_version_dataset_gallery").style(grid=4)
                    with gr.Row():
                        gr_project_version_dataset_label = gr.Textbox(f"Dataset: None", label='Dataset', lines=1, interactive = False)
            with gr.Column():
                # UI: update dataset
                with gr.Box():
                    # UI: upload dataset
                    with gr.Row():
                        # UI: dateset files uploader
                        input_train_data_set_files = gr.Files(interactive=True, label="Upload Dataset", type="file", tool="sketch")
                    with gr.Box():
                        # UI: dateset files upload post process
                        with gr.Row():
                            gr.Markdown(f"### Preprocess images")
                        with gr.Row():
                            # UI: dateset files upload post process width slider
                            process_width = gr.Slider(minimum=64, maximum=2048, step=8, label="Width", value=512, elem_id="process_width", interactive = True)
                            process_height = gr.Slider(minimum=64, maximum=2048, step=8, label="Height", value=512, elem_id="process_height", interactive = True)
                            preprocess_txt_action = gr.Dropdown(label='Existing Caption txt Action', value="ignore", choices=["ignore", "copy", "prepend", "append"], elem_id="preprocess_txt_action", interactive = True)
                        with gr.Row():
                            # UI: dateset images post process
                            with gr.Row():
                                # UI: dateset images post process functional
                                process_flip = gr.Checkbox(label='Create flipped copies', elem_id="process_flip", value=True, interactive = True)
                                process_split = gr.Checkbox(label='Split oversized images', elem_id="process_split", interactive = True)
                                process_focal_crop = gr.Checkbox(label='Auto focal point crop', elem_id="process_focal_crop", interactive = True)
                                process_multicrop = gr.Checkbox(label='Auto-sized crop', elem_id="process_multicrop", interactive = True)
                                process_caption = gr.Checkbox(label='Use BLIP for caption', elem_id="process_caption", interactive = True)
                                process_caption_deepbooru = gr.Checkbox(label='Use deepbooru for caption', visible=True, elem_id="process_caption_deepbooru", interactive = True)
                            with gr.Row(visible=False) as process_split_extra_row:
                                process_split_threshold = gr.Slider(label='Split image threshold', value=0.5, minimum=0.0, maximum=1.0, step=0.05, elem_id="process_split_threshold", interactive = True)
                                process_overlap_ratio = gr.Slider(label='Split image overlap ratio', value=0.2, minimum=0.0, maximum=0.9, step=0.05, elem_id="process_overlap_ratio", interactive = True)
                            with gr.Row(visible=False) as process_focal_crop_row:
                                process_focal_crop_face_weight = gr.Slider(label='Focal point face weight', value=0.9, minimum=0.0, maximum=1.0, step=0.05, elem_id="process_focal_crop_face_weight", interactive = True)
                                process_focal_crop_entropy_weight = gr.Slider(label='Focal point entropy weight', value=0.15, minimum=0.0, maximum=1.0, step=0.05, elem_id="process_focal_crop_entropy_weight", interactive = True)
                                process_focal_crop_edges_weight = gr.Slider(label='Focal point edges weight', value=0.5, minimum=0.0, maximum=1.0, step=0.05, elem_id="process_focal_crop_edges_weight", interactive = True)
                                process_focal_crop_debug = gr.Checkbox(label='Create debug image', elem_id="process_focal_crop_debug", interactive = True)
                            with gr.Column(visible=False) as process_multicrop_col:
                                gr.Markdown('Each image is center-cropped with an automatically chosen width and height.')
                                with gr.Row():
                                    process_multicrop_mindim = gr.Slider(minimum=64, maximum=2048, step=8, label="Dimension lower bound", value=384, elem_id="process_multicrop_mindim", interactive = True)
                                    process_multicrop_maxdim = gr.Slider(minimum=64, maximum=2048, step=8, label="Dimension upper bound", value=768, elem_id="process_multicrop_maxdim", interactive = True)
                                with gr.Row():
                                    process_multicrop_minarea = gr.Slider(minimum=64*64, maximum=2048*2048, step=1, label="Area lower bound", value=64*64, elem_id="process_multicrop_minarea", interactive = True)
                                    process_multicrop_maxarea = gr.Slider(minimum=64*64, maximum=2048*2048, step=1, label="Area upper bound", value=640*640, elem_id="process_multicrop_maxarea", interactive = True)
                                with gr.Row():
                                    process_multicrop_objective = gr.Radio(["Maximize area", "Minimize error"], value="Maximize area", label="Resizing objective", elem_id="process_multicrop_objective", interactive = True)
                                    process_multicrop_threshold = gr.Slider(minimum=0, maximum=1, step=0.01, label="Error threshold", value=0.1, elem_id="process_multicrop_threshold", interactive = True)
                        with gr.Row():
                            # UI: dataset global config
                            train_num_repetitions = gr.Number(value=-1, label="Train number of repetitions", elem_id="train_num_repetitions", interactive = True)
                    with gr.Row():
                        with gr.Row(elem_id=f"gr_project_version_dataset_gallery_container"):
                            # UI: dateset update button
                            update_dataset_btn = gr.Button(value="Update Dataset", variant="primary", elem_id=f'update_dataset_btn')
                        with gr.Row():
                            # UI: dateset update log
                            update_dataset_log = gr.HTML(elem_id=f'html_log_Update_Dataset')
        with gr.Row(visible=False) as train_row:
            # UI: train
            with gr.Box():
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            tiles = shared.list_checkpoint_tiles()
                            train_base_model = gr.Dropdown(label="Train base model",value= tiles[0] if len(tiles) > 0 else "", choices= tiles, interactive = True, elem_id="train_base_model")
                            train_base_model_refresh_button = ui.ToolButton(value=ui.refresh_symbol, elem_id="train_base_model_refresh_button")
                        with gr.Row():
                            train_base_on_sd_v2 = gr.Checkbox(label="Base on Stable Diffusion V2", value=False, elem_id="train_base_on_sd_v2", interactive = True)
                            use_sdxl = gr.Checkbox(label="Base on Stable Diffusion XL", value=False, elem_id="use_sdxl", interactive = True)
                        with gr.Row():
                            train_xformers = gr.Checkbox(label="Use xformers", value=True, elem_id="train_xformers", interactive = True)
                        with gr.Row():
                            train_clip_skip = gr.Number(label="Clip skip (2 if training anime model)", value=2, elem_id="train_clip_skip", interactive = True)
                        with gr.Row():
                            train_save_every_n_epochs = gr.Number(value=5, label="Save every n epochs", elem_id="train_save_every_n_epochs", interactive = True)
                    with gr.Column():
                        train_batch_size = gr.Number(value=1, label="Batch size", elem_id="train_batch_size", interactive = True)
                        train_num_epochs = gr.Number(value=40, label="Number of epochs", elem_id="train_num_epochs", interactive = True)
                        train_learning_rate = gr.Textbox(value="0.0001", label="Learning rate(Multi-select e.g. 0.0001,0.0002)", elem_id="train_learning_rate", interactive = True)
                        sd_script_args = gr.Textbox(value="", label="Append or override the sd_script args. (e.g. `--lr_scheduler=\"constant_with_warmup\" --max_grad_norm=0.0`)", elem_id="sd_script_args", interactive = True)
                    with gr.Column():
                        train_net_dim = gr.Number(value=128, label="Net dim (128 ~ 144MB)", elem_id="train_net_dim", interactive = True)
                        train_alpha = gr.Number(value=64, label="Alpha (default is half of Net dim)", elem_id="train_alpha", interactive = True)
                        train_optimizer_type = gr.Dropdown(label="Optimizer type(Multi-select)",value=["Lion"], choices=["Adam", "AdamW", "AdamW8bit", "Lion", "SGDNesterov", "SGDNesterov8bit", "DAdaptation", "AdaFactor"], multiselect = True, interactive = True, elem_id="train_optimizer_type")
                        train_mixed_precision = gr.Dropdown(label="Mixed precision (If your graphics card supports bf16 better)",value="fp16", choices=["fp16", "bf16"], interactive = True, elem_id="train_mixed_precision")
                with gr.Row():
                    with gr.Column(scale=2):
                        train_begin_log = gr.HTML(elem_id=f'html_log_Begin_Train')
                    with gr.Column():
                        with gr.Row():
                            train_finish_generate_all_checkpoint_preview=gr.Checkbox(label="Generate all checkpoint preview after train finished", value=True, elem_id="train_finish_generate_all_checkpoint_preview", interactive = True)
                        with gr.Row(elem_id=f"train_begin_btn_container"):
                            # UI: train button
                            train_begin_btn = gr.Button(value="Begin train", variant="primary", elem_id=f'begin_train_btn')
        with gr.Box(visible=False) as preview_box:
            # UI: train checkpoints
            with gr.Box():
                # UI: preview config
                with gr.Row():
                    preview_txt2img_prompt = gr.Textbox(label='Prompt', value="best quality,Amazing,finely detail,extremely detailed CG unity 8k wallpaper",placeholder="Prompt", elem_id="preview_txt2img_prompt", interactive = True, lines=3)
                with gr.Row():
                    preview_txt2img_negative_prompt = gr.Textbox(label='Negative Prompt', value="worst quality, low quality, normal quality",placeholder="Negative Prompt", elem_id="preview_txt2img_negative_prompt", interactive = True, lines=3)
                with gr.Row():
                    with gr.Column():
                        preview_width = gr.Slider(minimum=64, maximum=2048, step=8, label="Width", value=512, elem_id="preview_width", interactive = True)
                        preview_height = gr.Slider(minimum=64, maximum=2048, step=8, label="Height", value=512, elem_id="preview_height", interactive = True)
                        preview_batch_count = gr.Slider(minimum=1, step=1, label='Batch count', value=1, elem_id="preview_batch_count", interactive = True)
                        preview_batch_size = gr.Slider(minimum=1, step=1, label='Batch size', value=1, elem_id="preview_batch_size", interactive = True)
                    with gr.Column():
                        preview_sampling_method = gr.Dropdown(label='Sampling method', choices=[x.name for x in sd_samplers.samplers], value=[sd_samplers.samplers[0].name],multiselect = True, elem_id="preview_sampling_method", interactive = True)
                        preview_sampling_steps = gr.Textbox(label="Sampling steps", value="28",placeholder="like 20,24,28", elem_id="preview_sampling_steps", interactive = True)
                        preview_cfg_scale = gr.Textbox(label="CFG Scale Combination", value="10",placeholder="like 8,9,10,11", elem_id="preview_cfg_scale", interactive = True)
                        preview_seed = gr.Textbox(label="Seed Combination", value="-1",placeholder="like -1,-1,-1,-1", elem_id="preview_seed", interactive = True)
                        preview_lora_multiplier = gr.Textbox(label="Lora multiplier", value="0.6,0.7,0.8",placeholder="like 0.6,0.7,0.8", elem_id="preview_lora_multiplier", interactive = True)
                    with gr.Column():
                        preview_include_sub_img = gr.Checkbox(label="Include sub images", value=False,elem_id="preview_include_sub_img", interactive = True)
                        preview_generate_all_preview_log = gr.HTML(elem_id=f'html_log_Generate_all_preview')
                with gr.Row():
                    # UI: preview refresh button
                    preview_delete_all_btn = gr.Button(value="Delete all preview image", variant="primary")
                    preview_refresh_btn = gr.Button(value="Refresh all checkpoint preview info", variant="primary")
                    with gr.Row(elem_id=f"preview_generate_all_preview_btn_container"):
                        preview_generate_all_preview_btn = gr.Button(value="Generate all checkpoint preview", variant="primary", elem_id=f'preview_generate_all_preview_btn')
            with gr.Box():
                with gr.Row():
                    gr_trains_dropdown = gr.Dropdown([], label=f"Trains", value=None, interactive=True)
                    project_version_trains_refresh_button = ui.ToolButton(value=ui.refresh_symbol, elem_id="project_version_trains_refresh_button")
            for i in range(0, max_list_checkpoint):
                with gr.Box(visible = False) as train_checkpoint_row:
                    with gr.Row():
                        # UI: train checkpoints item
                        with gr.Column():
                            # UI: train checkpoints item info
                            with gr.Row():
                                train_checkpoint_info_name_list.append(gr.Textbox(label="Checkpoint name", value=None, interactive=False))
                            with gr.Row():
                                train_checkpoint_info_path_list.append(gr.Textbox(label="Checkpoint path", value=None, interactive=False, lines=3))
                        with gr.Column(scale=2):
                            # UI: train checkpoints item preview action
                            with gr.Row():
                                train_checkpoint_txt2txt_preview_gallery_list.append(gr.Gallery(value = None, visible= False, label='Output', show_label=False, elem_id=f"train_checkpoint_txt2txt_preview_gallery").style(grid=[4]))
                                train_checkpoint_txt2txt_preview_single_image_list.append(gr.Image(value = None, visible=False, label='Output', show_label=False, elem_id=f"txt2txt_image", interactive=False))
                            with gr.Row():
                                train_checkpoint_txt2txt_preview_btn_log_list.append(gr.HTML(elem_id=f'html_log_Preview'))
                            with gr.Row():
                                train_checkpoint_txt2txt_preview_delete_before_generate_list.append(gr.Checkbox(label="Delete preview images before generate", value=True, interactive = True))
                                train_checkpoint_txt2txt_preview_btn_list.append(gr.Button(value="Generate preview", variant="primary"))
                train_checkpoint_row_list.append(train_checkpoint_row)
        def checkpoint_box_outputs():
            return train_checkpoint_row_list + train_checkpoint_info_name_list + train_checkpoint_info_path_list + train_checkpoint_txt2txt_preview_gallery_list + train_checkpoint_txt2txt_preview_single_image_list
        def trains_area_outputs():
            return [gr_trains_dropdown] + checkpoint_box_outputs()
        def dataset_outputs():
            return [gr_project_version_dateset_row, train_row, preview_box, gr_project_version_dataset_gallery, gr_project_version_dataset_label]
        def dataset_config_inputs():
            return [
                train_num_repetitions,
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
            ]
        def train_config_inputs(): 
            return [
                train_base_model,
                train_batch_size,
                train_num_epochs,
                train_save_every_n_epochs,
                train_finish_generate_all_checkpoint_preview,
                train_optimizer_type,
                train_learning_rate,
                sd_script_args,
                train_net_dim,
                train_alpha,
                train_clip_skip,
                train_mixed_precision,
                train_xformers,
                train_base_on_sd_v2,
                use_sdxl,
            ]
        def preview_config_inputs():
            return [
                # preview view config
                preview_include_sub_img,
                # txt2img
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
            ]
        def all_config_inputs():
            return dataset_config_inputs() + train_config_inputs() + preview_config_inputs()
        # ====Footer====
        gr.Markdown(f"<center>version:{version} author: liasece </center>")

        # ====Callbacks====
        # project
        gr_project_dropdown.change(
            fn=on_ui_change_project_click,
            inputs=[gr_project_dropdown],
            outputs=[gr_project_dropdown, gr_project_version_row, gr_version_dropdown]+dataset_outputs()+trains_area_outputs()+all_config_inputs(),
        )
        project_refresh_button.click(
            fn = ui_refresh_project,
            inputs = [gr_project_dropdown, gr_version_dropdown],
            outputs = [gr_project_dropdown, gr_project_version_row, gr_version_dropdown]+dataset_outputs()+trains_area_outputs()+all_config_inputs(), 
        )
        create_project_btn.click(
            fn=on_ui_create_project_click,
            _js="ask_for_project_name",
            inputs=dummy_component,
            outputs=[gr_project_dropdown, gr_project_version_row, gr_version_dropdown]+dataset_outputs()+trains_area_outputs()+all_config_inputs(),
        )
        # project version
        gr_version_dropdown.change(
            fn=on_ui_change_project_version_click,
            inputs=[gr_project_dropdown, gr_version_dropdown],
            outputs=[gr_version_dropdown]+dataset_outputs()+trains_area_outputs()+all_config_inputs(),
        )
        project_version_refresh_button.click(
            fn = ui_refresh_version,
            inputs = [gr_project_dropdown, gr_version_dropdown],
            outputs = [gr_version_dropdown]+dataset_outputs()+trains_area_outputs()+all_config_inputs(), 
        )
        create_project_version_btn.click(
            fn=on_ui_create_project_version_click,
            _js="ask_for_project_version_name",
            inputs=[gr_project_dropdown, dummy_component],
            outputs=[gr_version_dropdown, gr_project_version_dateset_row]+trains_area_outputs()+all_config_inputs(),
        )
        # dataset
        process_split.change(
            fn=lambda show: gr_show(show),
            inputs=[process_split],
            outputs=[process_split_extra_row],
        )
        process_focal_crop.change(
            fn=lambda show: gr_show(show),
            inputs=[process_focal_crop],
            outputs=[process_focal_crop_row],
        )
        process_multicrop.change(
            fn=lambda show: gr_show(show),
            inputs=[process_multicrop],
            outputs=[process_multicrop_col],
        )
        update_dataset_btn.click(
            fn=wrap_gradio_gpu_call(on_ui_update_dataset_click, extra_outputs=[None]*len(dataset_outputs())+[""]),
            _js="on_ui_update_dataset_click",
            inputs=[
                dummy_component,
                gr_project_dropdown,
                gr_version_dropdown,
                input_train_data_set_files,
                # dataset images preprocess
            ]
                +dataset_config_inputs(),
            outputs=dataset_outputs()+[update_dataset_log],
            show_progress=False,
        )
        # train
        train_base_model_refresh_button.click(
            fn = ui_refresh_train_base_model,
            outputs = [train_base_model], 
        )
        train_begin_btn.click(
            fn=wrap_gradio_gpu_call(on_train_begin_click, extra_outputs=[None]*len(trains_area_outputs())+[""]),
            _js="on_train_begin_click",
            inputs=[
                dummy_component,
                gr_project_dropdown,
                gr_version_dropdown,
            ]
                # train config
                +train_config_inputs()
                # preview view config
                +preview_config_inputs()
            ,
            outputs=[train_begin_btn]+trains_area_outputs()+[train_begin_log],
        )
        # trains area  
        gr_trains_dropdown.change(
            fn=on_ui_change_project_version_trains_click,
            inputs=[gr_project_dropdown, gr_version_dropdown, gr_trains_dropdown],
            outputs=trains_area_outputs(),
        )
        project_version_trains_refresh_button.click(
            fn = on_ui_refresh_project_version_trains_click,
            inputs = [gr_project_dropdown, gr_version_dropdown, gr_trains_dropdown],
            outputs = trains_area_outputs(), 
        )
        # preview
        preview_delete_all_btn.click(
            fn=on_ui_delete_all_preview_images_click,
            inputs=[
                gr_project_dropdown, 
                gr_version_dropdown, 
                gr_trains_dropdown,
            ],
            outputs=trains_area_outputs(),
        )
        preview_refresh_btn.click(
            fn=gr_update_trains_area_list,
            inputs=[
                gr_project_dropdown, 
                gr_version_dropdown,
                gr_trains_dropdown,
            ],
            outputs=trains_area_outputs(),
        )
        preview_generate_all_preview_btn.click(
            fn=wrap_gradio_gpu_call(on_ui_preview_generate_all_preview_btn_click, extra_outputs=[None]*len(trains_area_outputs())+[""]),
            _js="on_ui_preview_generate_all_preview_btn_click",
            inputs=[
                dummy_component,
                gr_project_dropdown, 
                gr_version_dropdown, 
                gr_trains_dropdown,
                # preview view config
            ]
                +preview_config_inputs()
            ,
            outputs=trains_area_outputs()+[preview_generate_all_preview_log],
        )
        for i in range(0, max_list_checkpoint):
            # preview button callback
            train_checkpoint_txt2txt_preview_btn_list[i].click(
                fn=wrap_gradio_gpu_call(on_ui_preview_1_checkpoint_click, extra_outputs=[None,None,""]),
                inputs=[
                    gr_project_dropdown, 
                    gr_version_dropdown, 
                    gr_trains_dropdown,
                    train_checkpoint_info_name_list[i], 
                    train_checkpoint_info_path_list[i],
                    train_checkpoint_txt2txt_preview_delete_before_generate_list[i],
                    # preview view config
                ]
                    +preview_config_inputs()
                ,
                outputs=[
                    train_checkpoint_txt2txt_preview_gallery_list[i],
                    train_checkpoint_txt2txt_preview_single_image_list[i],
                    train_checkpoint_txt2txt_preview_btn_log_list[i],
                ],
            )
        train_tools.load(
            fn = ui_refresh_project,
            inputs = [gr_project_dropdown, gr_version_dropdown],
            outputs = [gr_project_dropdown, gr_project_version_row, gr_version_dropdown]+dataset_outputs()+trains_area_outputs()+all_config_inputs(), 
        )
    return train_tools

def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}

def create_refresh_button(outputs, refresh_method, refreshed_args, elem_id, inputs):
    def refresh(*args):
        refresh_method()
        args = refreshed_args(*args) if callable(refreshed_args) else refreshed_args
        return args
    refresh_button = ui.ToolButton(value=ui.refresh_symbol, elem_id=elem_id)
    refresh_button.click(
        fn=refresh,
        inputs=inputs,
        outputs=outputs
    )
    return refresh_button
