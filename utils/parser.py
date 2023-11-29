import os
import yaml
import argparse

class YamlParser:
    def __init__(self, file):
        """
        *file: .yaml file
        """
        self.file = file
        self.args = None

        self._parse()
        self._convert()

    def _parse(self):
        with open(self.file, 'r') as fd:
            self.args = yaml.load(fd, Loader=yaml.FullLoader)

    def _convert(self):
        """
        Convert the dictionary to argparse.Namespace
        """
        self.args = argparse.Namespace(**self.args)

    def add_args(self, **new_args):
        """
        Add new arguments to the parser
        """
        for k, v in new_args.items():
            setattr(self.args, k, v)

class ArgParser:
    def __init__(self, args):
        NotImplementedError

    def _parse(self):
        NotImplementedError

    def _convert(self):
        NotImplementedError

    def add_args(self, **new_args):
        NotImplementedError