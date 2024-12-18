import pandas as pd

class results():
    """Class for saving results to database, and subsequent access
    """

    db = ''

    def __init__(self, exp_obj, config, db_name, results_folder=''):
        self.ex = exp_obj # pass in experiment object
        self.config = config # passed in from from config file

        if results_folder:
            self.results_folder = results_folder
        elif 'results_folder' in self.ex.config['paths']:
            self.results_folder = self.ex.config['paths']['results_folder'] # Path()?
        else:
            self.results_folder = './'

        # open dataframe and save to object
        self.db = ''

        return

    def add(self, shot_dict, name, value, comment='', overwrite=True):
    
        # For now, just adding a seperate row for each value, that has a paired shot dict and name

        timestamp = ''

        return
    
    def get(self, shot_dict, name, info=False):
    
        value = ''
        timestamp = ''
        comment = ''

        if info:
            return value, timestamp, comment
        else:
            return value
    
    def delete(self, shot_dict, name):

        return
    
    def list_keys(self, shot_dict):
        
        # return all keys associated with a shot dictionary

        return