# -*- coding: UTF-8 -*-
import os
from PIL import Image

version = "0.0.1"

# print for debugging
def printD(*values: object):
    print(f"Train Tools:", *values)

# range all image in folder
def readImages(inputPath: str, level: int = 0, include_pre_level: bool = False, endswith :str | list[str] = [".png",".jpg",".jpeg",".bmp",".webp"], startswith: str | list[str] = None) -> list[Image.Image]:
    res = []
    paths = readImagePaths(inputPath, level, include_pre_level, endswith, startswith)
    for path in paths:
        img = Image.open(path)
        res.append(img)
    return res
    
def readImagePaths(inputPath: str, level: int = 0, include_pre_level: bool = False, endswith :str | list[str] = [".png",".jpg",".jpeg",".bmp",".webp"], startswith: str | list[str] = None) -> list[str]:
    res = []
    if not os.path.isdir(inputPath):
        return res

    file_path_list = []
    for file_name in os.listdir(inputPath):
        if level > 0:
            if os.path.isdir(os.path.join(inputPath, file_name)):
                res += readImagePaths(os.path.join(inputPath, file_name), level-1, include_pre_level, endswith, startswith)
        if level <= 0 or include_pre_level:
            ok = False
            if isinstance(endswith, list):
                for e in endswith:
                    if file_name.endswith(e):
                        ok = True
                        break
            else:
                if file_name.endswith(endswith):
                    ok = True
            if startswith is not None:
                startswith_ok = False
                if isinstance(startswith, list):
                    for e in startswith:
                        if file_name.startswith(e):
                            startswith_ok = True
                            break
                else:
                    if file_name.startswith(startswith):
                        startswith_ok = True
                ok = ok and startswith_ok
            if ok:
                file_path_list.append(file_name)
    file_path_list = sorted([x for x in file_path_list])
    for file_name in file_path_list:
        res.append(str(os.path.join(inputPath, file_name)))
    return res

def readPathSubDirNameList(path: str, level: int = 0) -> list[str]:
    res = []
    if not os.path.isdir(path):
        return res

    for file in os.listdir(path):
        if level > 0:
            if os.path.isdir(os.path.join(path, file)):
                res += readPathSubDirNameList(os.path.join(path, file), level-1)
        elif os.path.isdir(os.path.join(path, file)):
            res.append(file)
    return res

def readPathSubDirPathList(path: str, level: int = 0) -> list[str]:
    res = []
    if not os.path.isdir(path):
        return res

    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if level > 0:
            if os.path.isdir(file_path):
                res += readPathSubDirPathList(file_path, level-1)
        elif os.path.isdir(file_path):
            res.append(os.path.abspath(file_path))
    return res

# range all checkpoint in folder, [(name, path)]
def readCheckpoints(inputPath: str, level: int = 0) -> list[(str, str)]:
    res = []
    if not os.path.isdir(inputPath):
        return res

    file_path_list = []
    for file in os.listdir(inputPath):
        if level > 0:
            if os.path.isdir(os.path.join(inputPath, file)):
                res += readCheckpoints(os.path.join(inputPath, file), level-1)
        elif file.endswith(".checkpoint") or file.endswith(".safetensors"):
            file_path_list.append((file, os.path.getmtime(os.path.join(inputPath, file))))
    file_path_list = sorted([x for x in file_path_list], key=lambda x: x[1], reverse=True)
    for file_p in file_path_list:
        file = file_p[0]
        res.append((os.path.splitext(os.path.basename(file))[0] ,os.path.join(inputPath, file)))
    return res

