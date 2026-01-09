import os
from LAMP.meta import meta
from LAMP.utils.io import load_file

class ApolloDB(meta):
    """Interface layer for Apollo control / shot logs
    """

    __version = 0.1
    __name = 'Apollo'

    def __init__(self, exp_obj, config):
        """Initiate parent base meta class to get all shared attributes and funcs"""
        super().__init__(exp_obj, config)

        # load the pickle file in 
        # This should be done now right? not during the information access...?
        db_filepath = os.path.join(self.data_folder, self.config['source'])
        self.db = load_file(db_filepath)

        return
    
    def get_value(self, pname, shot_dict):
        """Allow more that could be done here. Currently, if single value specified returns, that. Otherwise a dataframe?"""

        if pname not in self.db.keys():
            print(f'Apollo DB error; get_value(); no parameter: {pname}')
            return
        
        # this can be a list of parameters?
        # maybe it could apply to more than one shot dictionary... but that sounds complicated...
        # or maybe if it applies across a series, return the list of values?

        if ('date' not in shot_dict) or ('run' not in shot_dict):
            print(f'Apollo DB error; get_value(); Date and Run required in shot dict')
            return

        index_string = str(shot_dict['date']) + '/' + shot_dict['run']

        if 'burst' in shot_dict:
            if 'shotnum' in shot_dict:
                index_tuple = (index_string, shot_dict['burst'], shot_dict['shotnum'])
            else:
                # all shots in this burst
                index_tuple = (index_string, shot_dict['burst'])
        else:
            if 'shotnum' in shot_dict:
                # assuming burst 1
                index_tuple = (index_string, 1, shot_dict['shotnum'])
            else:
                # all bursts and shots
                index_tuple = (index_string)

        return self.db.loc[index_tuple, pname]
    
    def list_params(self):
        return self.db.keys()
    
# print(slog.tail())
# print(clog.loc['20241002/run017', :])
# print(clog.loc[('20241002/run017',1), 'jet_z'])
# print(clog.loc[('20241002/run017',1,5), 'jet_z'])