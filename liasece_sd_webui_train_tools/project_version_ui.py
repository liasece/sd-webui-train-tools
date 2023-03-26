# -*- coding: UTF-8 -*-

import gradio as gr
import os
from liasece_sd_webui_train_tools.util import *

from liasece_sd_webui_train_tools.project import *
# from liasece_sd_webui_train_tools.project_version_ui import *
# from liasece_sd_webui_train_tools.ui import *
from liasece_sd_webui_train_tools.checkpoint_preview_ui import *
from liasece_sd_webui_train_tools.dateset_ui import *
# from liasece_sd_webui_train_tools.train_ui import *

def on_ui_change_project_click(to_project: str):
    if to_project == "":
        return
    printD("on_ui_change_project_click:", to_project)
    project_version_list = get_project_version_list(to_project)
    default_project_version = project_version_list[0] if len(project_version_list) > 0 else ""
    project_version_train_list = get_project_version_trains_list(to_project, default_project_version)
    default_project_version_train = project_version_train_list[0] if len(project_version_train_list) > 0 else ""
    save_train_tools_config(TrainToolsConfig(to_project, default_project_version))
    return [
        gr.Dropdown.update(value=to_project),
        gr.Row.update(visible=True),
        gr.Dropdown.update(value=default_project_version, choices=project_version_list),
    ]+get_project_version_dataset_box_update(to_project, default_project_version)+gr_update_trains_area_list(to_project, default_project_version, default_project_version_train)+load_gr_update(to_project, default_project_version)

def ui_refresh_project(now_project: str, now_project_version: str):
    # load from file
    file_cfg = load_train_tools_config()
    if (file_cfg.select_project != now_project and now_project != "" and now_project != None) or (file_cfg.select_version != now_project_version and now_project_version != "" and now_project_version != None):
        if now_project and now_project != "" and now_project != None:
            file_cfg.select_project = now_project
            file_cfg.select_version = now_project_version
        elif now_project_version and now_project_version != "" and now_project_version != None:
            file_cfg.select_version = now_project_version
        save_train_tools_config(file_cfg)
    # choose the project
    project_list = get_project_list()
    to_project = file_cfg.select_project
    if not project_list.__contains__(to_project):
        to_project = project_list[0] if len(project_list) > 0 else ""
    # choose the version
    project_version_list = get_project_version_list(to_project)
    to_project_version = file_cfg.select_version
    if not project_version_list.__contains__(to_project_version):
        to_project_version = project_version_list[0] if len(project_version_list) > 0 else ""
    # save to file if changed
    if file_cfg.select_project != to_project or file_cfg.select_version != to_project_version:
        file_cfg.select_project = to_project
        file_cfg.select_version = to_project_version
        save_train_tools_config(file_cfg)
    printD("ui_refresh_project:", project_list)
    project_version_train_list = get_project_version_trains_list(to_project, to_project_version)
    default_project_version_train = project_version_train_list[0] if len(project_version_train_list) > 0 else ""
    return [
        gr.Dropdown.update(value=to_project, choices=project_list),
        gr.Row.update(visible=True),
        gr.Dropdown.update(value=to_project_version, choices=project_version_list),
    ]+get_project_version_dataset_box_update(to_project, to_project_version)+gr_update_trains_area_list(to_project, to_project_version, default_project_version_train)+load_gr_update(to_project, to_project_version)

def on_ui_create_project_click(new_project_name):
    """
    Parameters:
        new_project_name: The name of the project created. This should be of type str, but due to the limitations of _js in different gradio versions, this value may be an array. Our plugin needs to be compatible with multiple versions of webui, so we need to make some judgments here.
    """
    # if new_project_name is list, convert to str
    if isinstance(new_project_name, list):
        if len(new_project_name) > 0:
            new_project_name = new_project_name[0]
        else:
            new_project_name = ""
    if new_project_name == "" or new_project_name == None:
        return [None]*4
    # check the project name not exists
    printD("create project:", new_project_name)
    os.makedirs(os.path.join(str(get_root_path()), new_project_name))
    # save to file
    save_train_tools_config(TrainToolsConfig(new_project_name, ""))
    project_list = get_project_list()
    return [
        gr.Dropdown.update(visible=True, choices=project_list, value=new_project_name),
        gr.Row.update(visible=True),
        gr.Dropdown.update(visible=True, choices=get_project_version_list(new_project_name), value=""),
    ]+get_project_version_dataset_box_update(new_project_name, "")+gr_update_trains_area_list(new_project_name, "", "")+load_gr_update(new_project_name, "")

def on_ui_change_project_version_click(project: str, to_version: str):
    if project == "":
        return
    printD("on_ui_change_project_version_click:", project, to_version)
    save_train_tools_config(TrainToolsConfig(project, to_version))
    project_version_train_list = get_project_version_trains_list(project, to_version)
    default_project_version_train = project_version_train_list[0] if len(project_version_train_list) > 0 else ""
    return [
        gr.Dropdown.update(value=to_version),
    ]+get_project_version_dataset_box_update(project, to_version)+gr_update_trains_area_list(project, to_version, default_project_version_train)+load_gr_update(project, to_version)

def ui_refresh_version(project: str, now_version: str):
    project_version_list = get_project_version_list(project)
    to_project_version = now_version
    if to_project_version == "":
        to_project_version = project_version_list[0] if len(project_version_list) > 0 else ""
    save_train_tools_config(TrainToolsConfig(project, to_project_version))
    project_version_train_list = get_project_version_trains_list(project, now_version)
    default_project_version_train = project_version_train_list[0] if len(project_version_train_list) > 0 else ""
    return [
        gr.Dropdown.update(visible=True, choices=project_version_list, value=to_project_version),
    ]+get_project_version_dataset_box_update(project, to_project_version)+gr_update_trains_area_list(project, to_project_version, default_project_version_train)+load_gr_update(project, to_project_version)

def on_ui_create_project_version_click(project: str, new_version_name: str):
    if new_version_name == "":
        return [None]*3
    # check the version name not exists
    printD(project, "create version:", new_version_name)
    os.makedirs(os.path.join(get_project_version_root_path(project), new_version_name))
    version_list = get_project_version_list(project)
    save_train_tools_config(TrainToolsConfig(project, new_version_name))
    project_version_train_list = get_project_version_trains_list(project, new_version_name)
    default_project_version_train = project_version_train_list[0] if len(project_version_train_list) > 0 else ""
    return [
        gr.Dropdown.update(visible=True, choices=version_list, value=new_version_name),
        gr.Row.update(visible=True),
    ]+gr_update_trains_area_list(project, new_version_name, default_project_version_train)+load_gr_update(project, new_version_name)
