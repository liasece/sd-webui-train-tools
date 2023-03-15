# -*- coding: UTF-8 -*-

import sys
import os
import glob

def filePathToPythonModuleName(base_path: str, path: str) -> str:
    path = os.path.abspath(path)
    base_path = os.path.abspath(base_path)
    if not path.startswith(base_path):
        return ""
    res = path
    if res.startswith(base_path):
        res = res[len(base_path):]
    if res.endswith(".py"):
        res = res[:-3]
    if res.endswith("__init__"):
        res = res[:-8]
    while res.endswith(os.path.sep):
        res = res[:-len(os.path.sep)]
    while res.startswith(os.path.sep):
        res = res[len(os.path.sep):]
    res = res.replace(os.path.sep, ".")
    if res.endswith("__init__"):
        res = ""
    return res

def listAllDirPythonModules(base_path: str, path: str) -> list[str]:
    res = []
    for sub_path in os.listdir(path):
        sub_path = os.path.join(path, sub_path)
        if os.path.isdir(sub_path):
            li = listAllDirPythonModules(base_path, sub_path)
            module_name = filePathToPythonModuleName(base_path, sub_path)
            if len(li) > 0 and module_name != "":
                res += li
                res.append(module_name)
        elif sub_path.endswith(".py"):
            # python file
            module_name = filePathToPythonModuleName(base_path, sub_path)
            if module_name != "":
                res.append(module_name)
    return res

class PythonContextWarper:
    def __init__(self, to_module_path: str, path_include: list[str] | str = [], sub_module: list[str] | str = None):
        self.to_module_path = os.path.abspath(to_module_path)
        self.sys_modules_backup = dict()
        self.sys_path_backup = list()
        self.sub_module_list = list()
        self.path_include = list()
        if isinstance(path_include, str):
            self.path_include.append(os.path.abspath(path_include))
        else:
            for path in path_include:
                self.path_include.append(os.path.abspath(path))
        if sub_module is None:
            self.sub_module_list = listAllDirPythonModules(self.to_module_path, self.to_module_path)
        elif isinstance(sub_module, str):
            self.sub_module_list.append(sub_module)
        else:
            self.sub_module_list += sub_module

    def enter(self):
        self.sys_path_backup = sys.path
        self.sys_modules_backup = dict()
        for mod in self.sub_module_list:
            if mod in sys.modules:
                self.sys_modules_backup[mod] = sys.modules[mod]
                del sys.modules[mod]
        for path in self.path_include:
            sys.path.insert(0, path)
        sys.path.insert(0,self.to_module_path)
        return self

    def __enter__(self):
        return self.enter()

    def exit(self):
        sys.path = self.sys_path_backup
        for mod,value in self.sys_modules_backup.items():
            sys.modules[mod] = value

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit()
