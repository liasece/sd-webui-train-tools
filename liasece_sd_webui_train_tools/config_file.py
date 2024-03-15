import json
import gradio as gr

from liasece_sd_webui_train_tools.project import *

default_config = [
    # dataset
    {"train_num_repetitions": -1},
    {"process_width": 512},
    {"process_height": 512},
    {"preprocess_txt_action": "ignore"},
    {"process_flip": True},
    {"process_split": False},
    {"process_caption": False},
    {"process_caption_deepbooru": False},
    {"process_split_threshold": 0.5},
    {"process_overlap_ratio": 0.2},
    {"process_focal_crop": False},
    {"process_focal_crop_face_weight": 0.9},
    {"process_focal_crop_entropy_weight": 0.15},
    {"process_focal_crop_edges_weight": 0.5},
    {"process_focal_crop_debug": False},
    {"process_multicrop": False},
    {"process_multicrop_mindim": 384},
    {"process_multicrop_maxdim": 768},
    {"process_multicrop_minarea": 4096},
    {"process_multicrop_maxarea": 409600},
    {"process_multicrop_objective": "Maximize area"},
    {"process_multicrop_threshold": 0.1},
    # train
    {"train_base_model": None},
    {"train_batch_size": 1},
    {"train_num_epochs": 20},
    {"train_save_every_n_epochs": 2},
    {"train_finish_generate_all_checkpoint_preview": True},
    {"train_optimizer_type": ["Lion"]},
    {"train_learning_rate": "0.0001"},
    {"sd_script_args": ""},
    {"train_net_dim": 128},
    {"train_alpha": 64},
    {"train_clip_skip": 2},
    {"train_mixed_precision": "fp16"},
    {"train_xformers": True},
    {"train_base_on_sd_v2": False},
    {"use_sdxl": False},
    # preview
    {"preview_include_sub_img": False},
    {"preview_txt2img_prompt": "best quality,Amazing,finely detail,extremely detailed CG unity 8k wallpaper"},
    {"preview_txt2img_negative_prompt": "low quality"},
    {"preview_sampling_method": [
        "Euler a"
    ]},
    {"preview_sampling_steps": "28"},
    {"preview_width": 512},
    {"preview_height": 512},
    {"preview_batch_count": 1},
    {"preview_batch_size": 1},
    {"preview_cfg_scale": "10"},
    {"preview_seed": "-1"},
    {"preview_lora_multiplier": "0.6,0.7,0.8,1"},
]

def load_gr_update(project: str, version: str) -> list:
    tmp = []
    for item in default_config:
        tmp += [{
            "elem_id": list(item.keys())[0],
            "value": item[list(item.keys())[0]],
        }]
    load_dataset_config(project, version, tmp)
    load_train_config(project, version, tmp)
    load_preview_config(project, version, tmp)
    res = []
    for item in tmp:
        res += [ item["value"] ]
    return res

def load_config(path: str, to_list: list[dict]):
    """Load the config file from the given path. """
    # if the file doesn't exist, return an empty dict
    if not os.path.exists(path):
        return
    with open ( path , "r" ) as f :
        data = json.load( f )
        for k, v in data.items():
            for to in to_list:
                if to["elem_id"] == k:
                    to["value"] = v
                    break

def save_config(path: str, from_list: dict | list[dict]):
    """Save the config file to the given path. """
    # make sure the path exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {}
    if isinstance(from_list, dict):
        data = from_list
    else:
        for item in from_list:
            data[item["elem_id"]] = item["value"]
    with open ( path , "w" ) as f :
        json.dump(data, f, indent="\t")

def load_train_config(project: str, version: str, to_list: list):
    project_version_path = get_project_version_path(project, version)
    if project_version_path == None or project_version_path == "":
        return
    load_config(os.path.join(project_version_path, "train_config.json"), to_list)

def save_train_config(project: str, version: str, from_list: dict | list[dict]) -> None :
    project_version_path = get_project_version_path(project, version)
    if project_version_path == None or project_version_path == "":
        return
    save_config(os.path.join(project_version_path, "train_config.json"), from_list)

def load_dataset_config(project: str, version: str, to_list: list):
    project_version_path = get_project_version_path(project, version)
    if project_version_path == None or project_version_path == "":
        return
    load_config(os.path.join(project_version_path, "dataset_config.json"), to_list)

def save_dataset_config(project: str, version: str, from_list: dict | list[dict]) -> None :
    project_version_path = get_project_version_path(project, version)
    if project_version_path == None or project_version_path == "":
        return
    save_config(os.path.join(project_version_path, "dataset_config.json"), from_list)

def load_preview_config(project: str, version: str, to_list: list):
    project_version_path = get_project_version_path(project, version)
    if project_version_path == None or project_version_path == "":
        return
    load_config(os.path.join(project_version_path, "preview_config.json"), to_list)

def save_preview_config(project: str, version: str, from_list: dict | list[dict]) -> None :
    project_version_path = get_project_version_path(project, version)
    if project_version_path == None or project_version_path == "":
        return
    save_config(os.path.join(project_version_path, "preview_config.json"), from_list)

class TrainToolsConfig:
    def __init__(self, select_project: str = "", select_version: str = ""):
        self.select_project = select_project
        self.select_version = select_version

def load_train_tools_config() -> TrainToolsConfig:
    root_path = get_root_path()
    if root_path == None or root_path == "":
        return
    tmp = [
        {
            "elem_id": "select_project",
            "value": None,
        },
        {
            "elem_id": "select_version",
            "value": None,
        },
    ]
    load_config(os.path.join(root_path, "train_tools_config.json"), tmp)
    cfg = TrainToolsConfig()
    cfg.select_project = tmp[0]["value"]
    cfg.select_version = tmp[1]["value"]
    return cfg

def save_train_tools_config(cfg: TrainToolsConfig) :
    root_path = get_root_path()
    if root_path == None or root_path == "":
        return
    save_config(os.path.join(root_path, "train_tools_config.json"), cfg.__dict__)
