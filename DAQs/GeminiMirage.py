"""Interface layer for Gemini, using Mirage
"""

class DAQ:

    __version = 0.1
    __name = 'GeminiMirage'
    __authors = ['Brendan Kettle']

    def __init__(self, exp_obj):
        print(f'Using {self.__name} DAQ')
        # pass in experiment object
        self.ex = exp_obj
        self.data_folder = self.ex.config['paths']['data_folder']
        return
    
    def get_shot_data(self, shot_dict):


        return
