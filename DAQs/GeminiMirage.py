"""Interface layer for Gemini, using Mirage
"""
from .DAQ import DAQ

class GeminiMirage(DAQ):

    __version = 0.1
    __name = 'GeminiMirage'
    __authors = ['Brendan Kettle']

    def __init__(self, exp_obj):
        # pass in experiment object
        self.ex = exp_obj
        self.data_folder = self.ex.config['paths']['data_folder']
        # Initiate parent base DAQ class to get all shared attributes and funcs
        super().__init__()
        return

    def _build_shot_filepath(self, diagnostic, date, run, shotnum, ext):
        shot_path = f'{self.data_folder}/{diagnostic}/{date}/{run}/Shot{str(shotnum).zfill(3)}.{ext}'
        return shot_path
    
    # perhaps some of this can move to base class?
    def get_shot_data(self, diag_name, shot_dict):

        diag_config = self.ex.diags[diag_name].config['setup']
        required = ['data_folder','data_ext','data_type']
        for param in required:
            if param not in diag_config:
                print(f"get_shot_data() error: {self.__name} DAQ requires a config['setup'] parameter '{param}' for {diag_name}")
                return None
            
        # TO DO: OR can use GSN

        required = ['date','run','shotnum']
        for param in required:
            if param not in shot_dict:
                print(f"get_shot_data() error: {self.__name} DAQ requires a shot_dict['{param}'] value")
                return None

        shot_filepath = self._build_shot_filepath(diag_config['data_folder'], shot_dict['date'], shot_dict['run'], shot_dict['shotnum'], diag_config['data_ext'])

        if diag_config['data_type'] == 'image':
            shot_data = self.load_imdata(shot_filepath)
        else:
            print('Non-image data loading not yet supported... probably need to add text at least?')

        return shot_data
