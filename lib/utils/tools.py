import numpy as np
import os, sys
import pickle
import yaml
import json
try:
    from easydict import EasyDict as edict
except ImportError:
    # If easydict is not available, use a simple dictionary instead
    class edict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__ = self
from typing import Any, IO

ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')


class TextLogger:
    def __init__(self, log_path):
        self.log_path = log_path
        # Recommend specifying utf-8 for log writing to prevent garbled text
        with open(self.log_path, "w", encoding='utf-8') as f:
            f.write("")

    def log(self, log):
        # Recommend specifying utf-8 for log writing
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

    # Force utf-8 when reading included sub-configuration files
    with open(filename, 'r', encoding='utf-8') as f:
        if extension in ('yaml', 'yml'):
            return yaml.load(f, Loader)
        elif extension in ('json',):
            return json.load(f)
        else:
            return ''.join(f.readlines())


def get_config(config_path):
    yaml.add_constructor('!include', construct_include, Loader)
    # Force utf-8 when reading main configuration file (this is the core location causing the previous error)
    with open(config_path, 'r', encoding='utf-8') as stream:
        config = yaml.load(stream, Loader=Loader)
    config = edict(config)
    _, config_filename = os.path.split(config_path)
    config_name, _ = os.path.splitext(config_filename)
    config.name = config_name
    return config


def ensure_dir(path):
    """
    Create path by first checking its existence.
    :param path: path
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)


def read_pkl(data_url):
    with open(data_url, 'rb') as file:
        # Custom Unpickler to handle persistent_load errors
        class CustomUnpickler(pickle.Unpickler):
            def persistent_load(self, pid):
                # persistent IDs must be strings; if not, convert to string
                if isinstance(pid, str):
                    # If it is a string ID, try to return the corresponding data
                    # Here we return None because these data are usually not important in the current context
                    return None
                else:
                    # If not a string, convert to string and handle
                    return None

        try:
            # First attempt standard loading
            content = pickle.load(file, encoding='latin1')
        except Exception as e:
            try:
                # If fails, reset file pointer and use custom Unpickler
                file.seek(0)
                unpickler = CustomUnpickler(file)
                content = unpickler.load()
            except Exception as e2:
                # If still fails, try without encoding parameter
                file.seek(0)
                try:
                    content = pickle.load(file)
                except Exception as e3:
                    # Last attempt: use a different encoding
                    file.seek(0)
                    content = pickle.load(file, encoding='bytes')
    return content