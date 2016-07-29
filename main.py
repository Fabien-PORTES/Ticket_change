# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 09:25:39 2016

@author: fportes
"""
import os, re, subprocess
import sys, argparse, json

parser = argparse.ArgumentParser()
parser.add_argument("configuration_file_path", help="Path to the configuration file")

conf_path = parser.parse_args().configuration_file_path
if not os.path.isfile(conf_path):
    print("The path of the configuration file is wrong.\n%s\n does not exist." %conf_path)
    print("Execution aborted")
    sys.exit(1)

class fileConfig:
    def __init__(self, name):
        self.name = name
        self.params = dict()
    def fill_param(self, param_name, param_value):
        self.params[param_name] = param_value
    def show_conf(self):
        print(self.name)
        print(self.params)
    def get_name_param(self):
        args = json.dumps(self.params, separators=(',', ':'))
        return self.name, args

class Config:
    def __init__(self, path):
        self.path = os.path.normpath(path)
        self.scripts_list = []
    def fill(self):
        with open(self.path, 'r') as bloc_file:
            for line in bloc_file:
                line = line.strip()
                #print(line)
                if line == "" or '#' in line[0]:
                    continue
                elif re.match("\s*(?:(?:\-\-\>)|(?:\-\>))\s*", line):
                    script = re.match("\s*(?:(?:\-\-\>)|(?:\-\>)\s*)(.+)", line).groups()[0]
                    self.scripts_list.append(fileConfig(script.strip()))
                elif '.' in line[0]:
                    param_name, param_value = [x.strip() for x in line[1:].split('=')]
                    self.scripts_list[-1].fill_param(param_name, param_value)
    def show_conf(self):
        for file in self.scripts_list:
            file.show_param()
    def exec_config(self):
        for file in self.scripts_list:
            f, args = file.get_name_param()
            exit_code = subprocess.call([sys.executable, f, args])
            if exit_code != 0:
                print("The execution of the following script FAILED\nFile : \
                {filename}\nArgs :\n{argmts}".format(filename = f, argmts = args))
            else:
                print("The execution of the following script SUCCEEDED\nFile : \
                {filename}\nArgs :\n{argmts}".format(filename = f, argmts = args))
                
conf = Config(conf_path)
conf.fill()
conf.exec_config()







