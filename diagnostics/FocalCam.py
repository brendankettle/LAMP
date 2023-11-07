from .diagnostic import Diagnostic

class FocalCam(Diagnostic):

    __version = 0.1
    __authors = ['Brendan Kettle']
    __requirements = ''

    def __init__(self, exp_obj, config_filepath):
        print('Starting up FocalCam diagnostic')
        # ?? Get all shared attributes and funcs from base Diagnostic
        Diagnostic.__init__(self)
        return
