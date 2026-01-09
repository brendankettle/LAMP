import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
import time

def entries2columns(y):
    N = np.shape(y)
    if len(N)==1:
        y_reshaped = [[i] for i in y]
    else:
        y_reshaped = [[i[n] for i in y] for n in range(N[1])]
    return y_reshaped

class Results():
    """Class for saving results to database, and subsequent access
    Some stuff taken from Apollo here, but it operates differently.
    Might seem a bit inefficient, but I'm giving each results its own row.
    That way you don't have to add columns each time you add a result, so more flexible that way.
    But more importantly, each value can have a decription, and information like comments, who performed the analysis and when etc.
    """

    def __init__(self, exp_obj, config, db_name, results_folder=''):
        self.ex = exp_obj # pass in experiment object
        self.config = config # passed in from from config file

        if results_folder:
            self.results_folder = results_folder
        elif 'results_folder' in self.ex.config['paths']:
            self.results_folder = self.ex.config['paths']['results_folder'] # Path()?
        else:
            self.results_folder = './'

        # check pkl extension. NB; currently not .pickle, but assume .pkl
        if ('.pkl' in  db_name) or ('.pickle' in db_name):
            self.db_filename = db_name
        else:
            self.db_filename = f'{db_name}.pkl'
        self.db_name = db_name

        # check if database file exists, otherwise create a new one
        self.db_filepath = Path(os.path.join(self.results_folder,self.db_filename))
        if self.db_filepath.is_file():
            self.db = pd.read_pickle(self.db_filepath)
        else:
            # create new dataframe and save to object
            print(f'No results file found, creating new database (not yet saved): {self.db_filepath}')
            self.db = pd.DataFrame()

        return

    def save(self):
        # more checks?
        self.db.to_pickle(self.db_filepath)
        print(f'Results database saved: {self.db_filepath}') # Temp?
        return
        
    def add(self, name, value, shot_dict=None, description='', details='', user='', script='', overwrite=True, save=True):

        # if shot_dict is empty (''), make a blank dictionary, save as general data
        if shot_dict == '' or shot_dict is None: 
            multi_index_names = self.db.index.names # work out what is in shot dictionary
            shot_dict_names = multi_index_names[:-1]# last one is "name"
            shot_dict = {v:'' for v in shot_dict_names} # turn into dictionary
        indexes = self.make_index(shot_dict, name)

        if not script:
            script = sys.argv[0] # __file__

        data_dict = {'value': value, 'description': description, 'details': details, 'user': user, 'script': script, 'timestamp': time.time()}

        index_values = self.make_index_values(shot_dict, name)
        try: 
            self.db.loc[index_values]
        except:
            #print('Does not exist') # temp
            self.db  = pd.concat([self.db, pd.DataFrame([data_dict],index=indexes)])
        else:
            #print('Exists') # temp
            self.db.loc[index_values] = data_dict

        # save by default (turn off if you are adding lots of rows, but don't want to filewrite each time?)
        if save:
            self.db.to_pickle(self.db_filepath)

        return
    
    def get(self, name, shot_dict=None, info=False):
    
        # if shot_dict is empty (''), make a blank dictionary, assuming getting general data
        if shot_dict == '' or shot_dict is None: 
            multi_index_names = self.db.index.names # work out what is in shot dictionary
            shot_dict_names = multi_index_names[:-1]# last one is "name"
            shot_dict = {v:'' for v in shot_dict_names} # turn into dictionary

        # can this work? if missing an index, remove from this (temporary) copy of dataframe
        df = self.db
        # for dict_key in shot_dict:
        #     if shot_dict[dict_key] == '':
        #         print(f'No {dict_key} defined')
        #         df = df.drop(dict_key, axis=1, inplace=False)
        #         shot_dict = shot_dict.pop(dict_key)

        try:
            value = df['value'].loc[self.make_index_values(shot_dict, name)]
        except:
            print(f'Results.get() failed; invalid shot or name? : {shot_dict}, "{name}"')
            return False

        if info:
            description = df['description'].loc[self.make_index_values(shot_dict, name)]
            details = df['details'].loc[self.make_index_values(shot_dict, name)]
            user = df['user'].loc[self.make_index_values(shot_dict, name)]
            script = df['script'].loc[self.make_index_values(shot_dict, name)]
            timestamp = df['timestamp'].loc[self.make_index_values(shot_dict, name)]
            return value, description, details, user, script, timestamp
        else:
            return value
    
    def contents(self, shot_dict =''):

        # what values associated with given shot?
        if shot_dict != '':
            shot_df = self.get(shot_dict, '')
            names = shot_df.index.get_level_values('name').unique()
        # overall what unique values across database?
        else:
            names = self.db.index.get_level_values('name').unique()
        
        # also return number of shots? days etc.?

        return names.to_list()

    def shots(self):
        # return the unique shots in the database
        shot_df = self.db
        name_level = shot_df.index.nlevels - 1
        unique_shots = shot_df.index.droplevel(name_level).to_flat_index().unique().to_list()
        return unique_shots
    
    def delete(self, name, shot_dict=None):
        
        # if shot_dict is empty (''), make a blank dictionary, assuming getting general data
        if shot_dict == '' or shot_dict is None: 
            multi_index_names = self.db.index.names # work out what is in shot dictionary
            shot_dict_names = multi_index_names[:-1]# last one is "name"
            shot_dict = {v:'' for v in shot_dict_names} # turn into dictionary

        # print(self.make_index_values(shot_dict, name))
        self.db = self.db.drop(index=self.make_index_values(shot_dict, name))
        #print(self.db.head())
        return
    
    def make_index_names(self, shot_dict, name):
        index_names = list(shot_dict.keys())
        if name != '':
            index_names.append('name')
        return tuple(index_names)

    def make_index_values(self, shot_dict, name):
        index_values = list(shot_dict.values())
        if name != '':
            index_values.append(name)
        return tuple(index_values)
    
    def make_index(self, shot_dict, name):
        index_names = self.make_index_names(shot_dict, name)
        index_values = self.make_index_values(shot_dict, name)
        return pd.MultiIndex.from_arrays(entries2columns(list(index_values)),names=index_names)