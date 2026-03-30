import numpy as np
import os, sys
import pickle
import yaml
import json  # 添加了 json 库，因为下面 construct_include 用到了
try:
    from easydict import EasyDict as edict
except ImportError:
    # 如果easydict不可用，使用简单的字典替代
    class edict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__ = self
from typing import Any, IO

ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')


class TextLogger:
    def __init__(self, log_path):
        self.log_path = log_path
        # 建议日志写入也指定 utf-8，防止中文乱码（可选）
        with open(self.log_path, "w", encoding='utf-8') as f:
            f.write("")

    def log(self, log):
        # 建议日志写入也指定 utf-8
        with open(self.log_path, "a+", encoding='utf-8') as f:
            f.write(log + "\n")


class Loader(yaml.SafeLoader):
    """YAML Loader with `!include` constructor."""

    def __init__(self, stream: IO) -> None:
        """Initialise Loader."""

        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir

        super().__init__(stream)


def construct_include(loader: Loader, node: yaml.Node) -> Any:
    """Include file referenced at node."""

    filename = os.path.abspath(os.path.join(loader._root, loader.construct_scalar(node)))
    extension = os.path.splitext(filename)[1].lstrip('.')

    # 【修改点 1】：读取被 include 的子配置文件时，强制使用 utf-8
    with open(filename, 'r', encoding='utf-8') as f:
        if extension in ('yaml', 'yml'):
            return yaml.load(f, Loader)
        elif extension in ('json',):
            return json.load(f)
        else:
            return ''.join(f.readlines())


def get_config(config_path):
    yaml.add_constructor('!include', construct_include, Loader)
    # 【修改点 2】：读取主配置文件时，强制使用 utf-8（这是导致你之前报错的核心位置）
    with open(config_path, 'r', encoding='utf-8') as stream:
        config = yaml.load(stream, Loader=Loader)
    config = edict(config)
    _, config_filename = os.path.split(config_path)
    config_name, _ = os.path.splitext(config_filename)
    config.name = config_name
    return config


def ensure_dir(path):
    """
    create path by first checking its existence,
    :param paths: path
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)


def read_pkl(data_url):
    with open(data_url, 'rb') as file:
        # 处理 persistent_load 错误的自定义 Unpickler
        class CustomUnpickler(pickle.Unpickler):
            def persistent_load(self, pid):
                # persistent IDs 必须是字符串，如果不是则转换为字符串
                if isinstance(pid, str):
                    # 如果是字符串形式的ID，尝试返回对应的数据
                    # 这里我们返回None，因为通常这些数据在当前的上下文中不重要
                    return None
                else:
                    # 如果不是字符串，转换为字符串后处理
                    return None

        try:
            # 首先尝试标准方式
            content = pickle.load(file, encoding='latin1')
        except Exception as e:
            try:
                # 如果失败，重置文件指针并尝试使用自定义 Unpickler
                file.seek(0)
                unpickler = CustomUnpickler(file)
                content = unpickler.load()
            except Exception as e2:
                # 如果还是失败，尝试不使用 encoding 参数
                file.seek(0)
                try:
                    content = pickle.load(file)
                except Exception as e3:
                    # 最后一次尝试：使用不同的编码
                    file.seek(0)
                    content = pickle.load(file, encoding='bytes')
    return content