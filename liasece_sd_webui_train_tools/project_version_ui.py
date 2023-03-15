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
    project_version_list = load_project_version_list(to_project)
    default_project_version = project_version_list[0] if len(project_version_list) > 0 else ""
    return [
        gr.Dropdown.update(value=to_project),
        gr.Row.update(visible=True),
        gr.Dropdown.update(value=default_project_version, choices=project_version_list),
    ]+get_project_version_dataset_box_update(to_project, default_project_version)+gr_update_checkpoint_list(to_project, default_project_version)

def ui_refresh_project(now_project: str, now_project_version: str):
    project_list = load_project_list()
    default_project = now_project
    if default_project == "":
        default_project = project_list[0] if len(project_list) > 0 else ""
    project_version_list = load_project_version_list(default_project)
    default_project_version = now_project_version
    if not project_version_list.__contains__(default_project_version):
        default_project_version = project_version_list[0] if len(project_version_list) > 0 else ""
    printD("ui_refresh_project:", project_list)
    return [
        gr.Dropdown.update(value=default_project),
        gr.Row.update(visible=True),
        gr.Dropdown.update(value=default_project_version, choices=project_version_list),
    ]+get_project_version_dataset_box_update(default_project, default_project_version)+gr_update_checkpoint_list(default_project, default_project_version)

def on_ui_create_project_click(new_project_name: str):
    if new_project_name == "" or new_project_name == None:
        return [None]*4
    # check the project name not exists
    printD("create project:", new_project_name)
    os.makedirs(os.path.join(str(save_project_path), new_project_name))
    project_list = load_project_list()
    return [
        gr.Dropdown.update(visible=True, choices=project_list, value=new_project_name),
        gr.Row.update(visible=True),
        gr.Dropdown.update(visible=True, choices=load_project_version_list(new_project_name), value=""),
    ]+get_project_version_dataset_box_update(new_project_name, "")+gr_update_checkpoint_list(new_project_name, "")

def on_ui_change_project_version_click(project: str, to_version: str):
    if project == "":
        return
    printD("on_ui_change_project_version_click:", project, to_version)
    return [
        gr.Dropdown.update(value=to_version),
    ]+get_project_version_dataset_box_update(project, to_version)+gr_update_checkpoint_list(project, to_version)

def ui_refresh_version(project: str, now_version: str):
    project_version_list = load_project_version_list(project)
    default_project_version = now_version
    if default_project_version == "":
        default_project_version = project_version_list[0] if len(project_version_list) > 0 else ""
    return [
        gr.Dropdown.update(visible=True, choices=project_version_list, value=default_project_version),
    ]+get_project_version_dataset_box_update(project, default_project_version)+gr_update_checkpoint_list(project, default_project_version)

def on_ui_create_project_version_click(project: str, new_version_name: str):
    if new_version_name == "":
        return [None]*3
    # check the version name not exists
    printD(project, "create version:", new_version_name)
    os.makedirs(os.path.join(get_project_version_root_path(project), new_version_name))
    version_list = load_project_version_list(project)
    return [
        gr.Dropdown.update(visible=True, choices=version_list, value=new_version_name),
        gr.Row.update(visible=True),
    ]+gr_update_checkpoint_list(project, new_version_name)
