# -*- coding: utf-8 -*-
"""【中文说明】工具模块：AttrDict 让字典可以用属性方式访问（h.xxx），build_env 用于复制配置文件。"""
""" from https://github.com/jik876/hifi-gan """
import os
import shutil


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def build_env(config, config_name, path):
    t_path = os.path.join(path, config_name)
    if config != t_path:
        os.makedirs(path, exist_ok=True)
        shutil.copyfile(config, os.path.join(path, config_name))
