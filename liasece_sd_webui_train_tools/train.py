import os
import subprocess
import sys
import importlib

import pkg_resources

from liasece_sd_webui_train_tools.ArgsList import ArgStore
from liasece_sd_webui_train_tools.util import *
from modules import script_loading

import liasece_sd_webui_train_tools.sd_scripts.train_network as train_network
import liasece_sd_webui_train_tools.sd_scripts.sdxl_train_network as sdxl_train_network

import liasece_sd_webui_train_tools.PythonContextWarper as pc
import liasece_sd_webui_train_tools.util as util

try:
    import lion_pytorch
    import dadaptation
except ModuleNotFoundError as error:
    required = {"lion-pytorch", "dadaptation"}
    installed = {p.key for p in pkg_resources.working_set}
    missing = required - installed
    if missing:
        print("missing some requirements, installing...")
        python = sys.executable
        subprocess.check_call([python, "-m", "pip", "install", *missing], stdout=subprocess.DEVNULL)


def train(cfg: ArgStore) -> None:
    args = cfg.create_args()
    with pc.PythonContextWarper(
            to_module_path= os.path.abspath(os.path.join(os.path.dirname(__file__), "sd_scripts")), 
            path_include= os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), 
            sub_module=["library", "networks"],
        ):
        # begin training
        if cfg.use_sdxl:
            trainer = sdxl_train_network.SdxlNetworkTrainer()
        else:
            trainer = train_network.NetworkTrainer()
        printD("train begin", args)
        trainer.train(args)
