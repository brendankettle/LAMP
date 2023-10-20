import configparser
import importlib

class Experiment:

    def __init__(self, config_filepath):
        self.config_filepath = config_filepath

        config = configparser.ConfigParser()
        config.read(config_filepath)
        DAQ_module = 'LAMP.DAQs.' + config['setup']['DAQ']

        #try:
        DAQ_lib = importlib.import_module(DAQ_module)
        #except ImportError:
        Data = DAQ_lib.DAQ()
