# -*- coding: UTF-8 -*-
import os
from PIL import Image

version = "0.0.1"

# print for debugging
def printD(*values: object):
    print(f"Train Tools:", *values)

# range all image in folder
def readImages(inputPath: str, level: int = 0, endswith :str | list[str] = [".png",".jpg",".jpeg",".bmp",".webp"]) -> list[Image.Image]:
    res = []
    if not os.path.isdir(inputPath):
        return res

    file_path_list = []
    for file in os.listdir(inputPath):
        if level > 0:
            if os.path.isdir(os.path.join(inputPath, file)):
                res += readImages(os.path.join(inputPath, file), level-1)
        else:
            ok = False
            if isinstance(endswith, list):
                for e in endswith:
                    if file.endswith(e):
                        ok = True
                        break
            else:
                if file.endswith(endswith):
                    ok = True
            if ok:
                file_path_list.append(file)
    file_path_list = sorted([x for x in file_path_list])
    for file in file_path_list:
        img = Image.open(os.path.join(inputPath, file))
        res.append(img)
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

# range all checkpoint in folder, [(name, path)]
def readCheckpoints(inputPath: str, level: int = 0) -> list[(str, str)]:
    res = []
    if not os.path.isdir(inputPath):
        return res

    file_path_list = []
    for file in os.listdir(inputPath):
        if level > 0:
            if os.path.isdir(os.path.join(inputPath, file)):
                res += readImages(os.path.join(inputPath, file), level-1)
        elif file.endswith(".checkpoint") or file.endswith(".safetensors"):
            file_path_list.append((file, os.path.getmtime(os.path.join(inputPath, file))))
    file_path_list = sorted([x for x in file_path_list], key=lambda x: x[1], reverse=True)
    for file_p in file_path_list:
        file = file_p[0]
        res.append((os.path.splitext(os.path.basename(file))[0] ,os.path.join(inputPath, file)))
    return res

