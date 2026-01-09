## Necessary files/functions for Nist model
from scipy.interpolate import interp1d, pchip
from matplotlib.patches import Patch
import matplotlib.pyplot as plt 
import os 
import numpy as np 
import pandas as pd 
import glob 
import re 

class Engine():
    ''' 
    Engine Class for pyNIST Library

    Handles the communication to the local database of elements for the pyNIST material classes
    Requires specifying the desired libray between NIST and ESTAR.

    This can be pre-loaded for efficient operation but is otherwise unnecessary for end users 
    as Material class loads engine on demand.

    Series of helper functions handle the offset in energy require for non-unique values in database, 
    parse formulas into elemental consituent parts and effectively handle the atomic weight calculations 
    corresponding to fractional weights of materials.

    Warning: Lowercase element names are ignored in parse_formula function and will error the system 

    Intialisation Parameters
    ----------
    name - str - ESTAR or NIST
    
    Functions 
    ---------

    Automatic - Happens during Initialisation: 
        _generate_atomic_table - loads atomic weights, atomic numbers from database file
        _generate_material_list - generates list of materials in the database 
        skiprows, col values - pre-declared values to enable scraping of the database files
                             - can be varied by user via "_declare_new_columns" method

    Manual - Called by user:
        parse_forumla - seperates a given chemical composotion into constituent 
                        elements and number
        db_scraper - for material loads/interpolates the cross-sections for given 
                     energies, if energies are not declared it loads default values. 
                     Non-unique values are nudged by 0.1 eV to enable interpolation but 
                     preserve k-edge
        _declare_new_columns - by default the Engine returns the total cross-section for 
                               each element, this method switches the column read by the 
                               scraper. Call Engine._declare_new_columns? for more details
                        
    '''

    def __init__(self, name) -> None:
        assert (name == 'NIST') or (name == 'ESTAR') or (name == 'PSTAR')

        # set Engine Version
        self.name = name
        self.path = '/'.join(os.path.dirname(__file__).split('/')[:])
        self.db_path = self.path+f"/database/{self.name}/"

        # Configure Elemental Database
        self.db = {'names': ['atomic', 'material_list'],
                   'formats': [pd.DataFrame, list]}
        self.db['atomic'] = self._generate_atomic_table()
        self.db['material_list'] = self._generate_material_list()

        # Set Engine Specific Lables
        if self.name == 'NIST':
            self.skiprows = 3
            self.col = -1
        elif self.name == 'ESTAR':
            self.skiprows = 8
            self.col = -1
        elif self.name == 'PSTAR':
            self.skiprows = 8
            self.col = -1

    def _declare_new_columns(self, val):
        '''
        NIST AND ESTAR databases include additional crosssections.
        For NIST: 
        Col 0 - Energies
        Col 1 - Coherent Scatter (Thomson)
        Col 2 - Incoherent Scatter (Compton)
        Col 3 - Photoelectric Absorption 
        Col 4 - Nuclear Pair Production
        Col 5 - Electronic Pair Production
        Col 6 - Total (All) [DEFAULT]

        For ESTAR:
        Col 0 - Energies
        Col 1 - Collisional (MeV cm2/g)
        Col 2 - Radiative (MeV cm2/g)
        Col 3 - Total (MeV cm2/g)  [DEFAULT]

        For PSTAR:
        Col 0 - Energies
        Col 1 - Collisional (MeV cm2/g)
        Col 2 - Radiative (MeV cm2/g)
        Col 3 - Total (MeV cm2/g)  [DEFAULT]
        '''
        self.col = val

    def _generate_atomic_table(self):
        ''' 
        Return dataframe for atomic number and weight
        Includes symbol parsing frame as well
        '''
        df = pd.read_csv(f'{self.path}/database/atomic_weight.db', delimiter='\t', header=0,
                         usecols=[0, 1, 2, 3, 5], names=['Z', 'Symbol', 'Name', 'Weight', 'Density'],
                         dtype={
                             'Z': int,
                             'Symbol': str,
                             'Name': str,
                             'Weight': str,
                             'Density': float,
        })
        df['Weight'] = df['Weight'].apply(self._parse_value)

        return df

    def _generate_material_list(self):
        ''' 
        returns list of available elements in database to mitigate errors
        eventually redundant as the database is completed
        '''
        self.fileslist = glob.glob(self.db_path+'*.txt')
        return [f.split('_')[-1].split('.')[0] for f in self.fileslist]

    def _parse_value(self, val):
        '''
        try/except function to correct the formatting on values in the database 
        Weird bug work around 
        '''
        try:
            if '[' in val:
                return float(val[1:-1])
            else:
                f = val.split('.')
                # print(f)
                return float('.'.join([f[0], f[1][0]]))
        except:
            return val

    def parse_formula(self, formula: str) -> dict:
        """
        Parses a chemical formula and returns a dictionary mapping elements to their respective counts in the formula.
        Args:
        formula: A string representing a chemical formula.
        Returns:
        A dictionary mapping elements to their respective counts in the formula.
        """
        # Initialize a dictionary to store the element counts
        element_counts = {}

        # Use a regular expression to match elements and counts in the formula
        elements = re.findall(r'([A-Z][a-z]*|[A-Z])([0-9]*\.?[0-9]*)', formula)

        # Iterate over the elements and counts in the formula
        for element, count in elements:
            # If the element is not in the element_counts dictionary, add it with a count of 0
            if element not in element_counts:
                element_counts[element] = 0

            # If the count is not empty, add it to the count for the element
            if count:
                element_counts[element] += float(count)
            # If the count is empty, increment the count for the element by 1
            else:
                element_counts[element] += float(1)

        # Return the element counts dictionary
        return element_counts

    def nonuniquer(self, arr, offset=1e-7):
        '''
        Shifts the k-edge in material databases by 0.1 eV to enable interpolation at the edge
        '''
        result = arr.copy()  # Create a copy of the input array to work with
        seen_values = {}  # A dictionary to store the count of each value

        for i in range(len(result)):
            if result[i] in seen_values:
                new_value = result[i] + offset
                while new_value in seen_values or new_value in result:
                    # Check for duplicates in both seen values and the new array
                    new_value += offset
                seen_values[new_value] = 1
                result[i] = new_value
            else:
                seen_values[result[i]] = 1
        assert np.all(np.diff(result) > 0)
        return result

    def db_scraper(self, material, energies=None):
        '''
        Main database scraping tool. 
        Looks for each element in the database according to the material string and loads necessary cross-sections
        (Optional) Interpolates cross-sections over the desired energies.         
        '''

        if material.lower() == 'vac':
            if energies is None:
                print('Energies must be defined for vacuum transmission')
            else:
                elist = energies
                clist = np.ones((1, len(energies)))
                return elist, clist[0]
        else:
            data = np.loadtxt(f'{self.db_path}{self.name}_{material}.txt', skiprows=self.skiprows)
            elist = np.array(data[:, 0])
            clist = np.array(data[:, self.col])
            if energies is None:
                return elist, clist
            else:
                ein2 = self.nonuniquer(elist)
                c3 = pchip(ein2, clist)
                cout = c3(energies)
                return energies, cout
            
    def _raw_db_scraper(self,file_string,energies=None):
        ''' 
        Flexibile database scraper, used in testing to evaluate new cross-section 
        tables outside of the main database.

        Declare "file_string" as full path to desired file.
        '''
        data = np.loadtxt(
            f'{file_string}', skiprows=self.skiprows)
        elist = np.array(data[:, 0])
        clist = np.array(data[:, self.col])
        if energies is None:
            return elist, clist
        else:
            ein2 = self.nonuniquer(elist)
            c3 = pchip(ein2, clist)
            cout = c3(energies)
            return energies, cout


class Material():
    ''' 
    Material Class for pyNIST Library

    Computes the necessary cross-sections from the defined database for any arbitary 
    material composition. Materials should be declared as camel-case elements such as 
    "SiO2" or "Al" for Quartz and elemental Aluminium respectively

    Warning: Lowercase element names are ignored.

    Intialisation Parameters
    ----------
    material - str - Chemical composition of material
    density - float - g/cc
    energies - array or None - array of desired energies in MeV if undeclared uses database defaults
    engine - str or Engine - Engine declaration must be type: "str" or "Engine"
                             For most efficient use pass pre-loaded Engine class (about 2x as fast)


    Functions 
    ---------

    Automatic - Happens during Initialisation: 
        generateCrossSections - sets cross-section (sigma) and energies corresponding to the material
    Manual - Called by user:
        get_transmission - calculates the transmission through material for thickness in mm
        get_absorption - calculates the absorption in material for thickness in mm

    '''

    def __init__(self, material, density, energies=None, engine='NIST',genericName=None):
        # initialise or use engine
        assert (type(engine) == str) or (type(engine) == Engine)

        if type(engine) == str:
            self.engine = Engine(engine)
        else:
            self.engine = engine

        self.material = self.engine.parse_formula(material)
        self.density = density

        # BK added
        if len(self.material) == 0:
            print(f'Warning, pyNist, unrecognised material: {material}')

        ## handle energies = None case 
        # load energies of first element to use as base

        if energies is None:
            energies,_ = self.engine.db_scraper(list(self.material.keys())[0])

        self.generateCrossSections(self.material, energies)

        if genericName is None:
            self.materialstr = material
        else:
            assert (type(genericName)==str)
            self.materialstr = genericName            

    def _parse_compound(self, compound):
        element_pat = re.compile("([A-Z][a-z]?)(\d*)")
        return element_pat.findall(compound)

    def generateCrossSections(self, dict, energies):
        x = self.engine.db['atomic']
        sigma = np.zeros_like(energies)
        total_weight = 0
        for key, value in dict.items():
            total_weight += value * x[x['Symbol']
                                      == key]['Weight'].to_numpy()[0]

        for key, value in dict.items():
            e, c = self.engine.db_scraper(key, energies)
            fractional_weight = (
                value * x[x['Symbol'] == key]['Weight'].to_numpy()[0])/total_weight
            sigma[:] += c * fractional_weight

        self.energies = energies
        self.sigma = sigma
        if (self.engine.name == 'ESTAR') or (self.engine.name == 'PSTAR'):
            self._estarcorrection()

    def _estarcorrection(self):
        ''' correction for the estar units '''
        self.sigma = self.sigma/self.energies

    def get_transmission(self, thickness):
        ''' calculates the transmission of x-rays through a layer:
        * thickness must be in mm '''
        self.sigma[self.sigma<0] = -self.sigma[self.sigma<0] # bug fix for some errornaeous data, e.g. in Ni?
        return np.exp(-self.sigma*self.density*(thickness*0.1))

    def get_absorption(self, thickness):
        ''' calculates the absorption of x-rays through a layer:
        * thickness must be in mm '''
        return 1-self.get_transmission(thickness)
    
    def get_profile(self, thickness, nstep=1000):
        '''
        Returns the z profile deposition for radiation transmitting through
        Thickness in mm, nstep is number of elements calculated
        By default this is 1000 so the resolution is thickness/nstep.

        Solved for the full array, particles are tracked until they run out of energy

        '''
        # initialise seed energies 
        hold_energies = self.energies
        # initialise arrays
        track_map = np.zeros((len(self.energies),nstep))
        energy_map = np.zeros((len(self.energies), nstep))

        # set initial energy 
        energy_map[:,0] = self.energies

        for n in range(1,nstep):
            track = self.get_absorption(thickness/nstep)*self.energies
            delta_e = self.get_transmission(thickness/nstep)*self.energies
            self.energies = np.clip(delta_e,0,None)
            self.generateCrossSections(self.material, self.energies)

            ## update arrays
            energy_map[:,n] = self.energies
            track_map[:,n] = track

        # generate step positions for plotting
        zsteps = np.linspace(0,thickness,nstep)

        # reset material database:
        self.generateCrossSections(self.material,hold_energies)

        return (track_map,energy_map, zsteps)


class Scintillators(Material):
    '''
    Sub-class of Material for Scintillators
    Adds kappa value for light yield calculations
    
        Kappa in units of N_Photons/MeV

    Stores library of pre-defined materials, densities, and yeilds for simple implementation
    '''

    def __init__(self, material_str, density, energies, engine='NIST', kappa=1):
        super().__init__(material_str, density, energies, engine)
        self.kappa = kappa

    # def __init__(self, material_str:str=None, 
    #                    density:float=None, 
    #                    energies:np.array=None, 
    #                    kappa:float=None,
    #                    engine='NIST'):
        
    #     assert (energies is not None), "Energies must be defined to proceed"
    #     self.energies = energies
    #     self.engine = engine
    #     self.__assertion(material_str, density, kappa)

    #     if self.__predefined_scintillators(material_str):
    #         self.predefined(material_str)
    #     else: 
    #         super().__init__(material_str, density, energies, engine)
    #         self.kappa = kappa
    #         self.fname = material_str

    # def predefined(self,mat):
    #     Scintillators = self.__predefined_scintillators()
    #     ScintillatorKeys = list(Scintillators.keys())

    #     if mat in Scintillators.keys():
    #         self.__init__(material_str='Bi4Ge3O12',
    #                       density=Scintillators[mat][1],
    #                       energies=self.energies,
    #                       kappa=Scintillators[mat][2],
    #                       engine=self.engine)
    #         self.fname = mat
            
    #     elif mat in [s[0] for s in Scintillators.values()]:
    #         idx = np.squeeze(np.argwhere([mat == s[0]
    #                    for s in Scintillators.values()]))
            
    #         self.__init__(material_str=Scintillators[ScintillatorKeys[idx]][0],
    #                       density=Scintillators[ScintillatorKeys[idx]][1],
    #                       energies=self.energies,
    #                       kappa=Scintillators[ScintillatorKeys[idx]][2],
    #                       engine=self.engine)
            
    #         self.fname = Scintillators.keys[idx]

    #     else:
    #         assert False, "Failed material definition - You shouldn't see this error"
            
    # def __predefined_scintillators(self, material_str=None):
    #     Scintillators = {
    #         'BGO': ['Bi4Ge3O12',7.1,8],
    #         'LYSO': ['Lu1.8Y0.2SiO2',7.1,30],
    #         'CH': ['C10H8O4',1.3,8],
    #         'YAG': ['Y3Al5O12', 4.55, 35],
    #         'CsI':['CsI',4.85,54],
    #     }

    #     if material_str is None:
    #         return Scintillators
    #     else:
    #         if material_str in Scintillators.keys():
    #             return True
    #         elif material_str in [s[0] for s in Scintillators.values()]:
    #             return True
    #         else:
    #             return False

    # def __assertion(self,material_str, density, kappa):
    #     if (self.__predefined_scintillators(material_str) 
    #             or 
    #         all([material_str is not None,
    #              density is not None,
    #              kappa is not None])): 
    #         pass
    #     else:
    #         raise AssertionError('''Material name not pre-defined list''')



class System():
    '''
    Array of Material instances in a series
    Dedicated class to handle RCF and LAS style distributions

    Input can be defined as System array:
        [nist.Material, thickness, active_flag]

    Or by a series of System.add_layer() commands

    Main additional functions is the rewrite of get_profile 
    to handle cross-material trajectories
    '''

    def __init__(self, system_dict=None,verbose=False):
        assert(type(verbose) is bool)

        ## Sanitise system import
        assert (type(system_dict) is type(None)) or (type(system_dict) == dict)

        self.verbose=verbose
        if type(system_dict) is type(None):
            if self.verbose:
                print('System is empty, use System.add_layer()')
            self.system = {}
            self.count = 0
        else:
            if self.verbose:
                print('System loaded')
            self.system = system_dict
            self.count = len(self.system.keys())

        ## Assertion check for each layer
        if self.count:
            self._check_system()

    def _check_system(self):
        for i,s in enumerate(self.system.values()):
            assert(type(s[0]) == Material), f'Layer {i} is not type(Material) it is {type(s[0])}'

    def add_layer(self,material,thickness,active=1,index=None):
        if index is None:
            index = self.count
        self.system[f'A{index}'] = [material,thickness,active] 
        self.count = len(self.system.keys())

        if self.verbose:
            print(f'Added {material,thickness,active} to index {index}')

    def add_system(self,system,index=None):
        assert (type(system) == System)
        if index is None:
            index = self.count
        
        for s in system.system.keys():
            self.add_layer(*system.system[s])
        

    def get_profile(self, nstep=1000):
        '''
        Returns the z profile deposition for radiation transmitting through
        Thickness in mm, nstep is number of elements calculated
        By default this is 1000 so the resolution is thickness/nstep.

        Solved for the full array, particles are tracked until they run out of energy

        '''
        # set target thickness as back of array
        cumulative_thickness = np.cumsum([s[1] for s in self.system.values()])
        thickness = cumulative_thickness[-1]
        dz = thickness / nstep 

        # initialise seed energies
        hold_energies = self.system[f'A0'][0].energies

        # initialise arrays
        track_map = np.zeros((len(self.system[f'A0'][0].energies), nstep))
        energy_map = np.zeros((len(self.system[f'A0'][0].energies), nstep))

        # set initial energy
        energy_map[:, 0] = self.system[f'A0'][0].energies

        for n in range(1, nstep):
            current_zposition = n*dz 
            loc = np.nonzero(current_zposition < cumulative_thickness)[0][0]

            # calculate track loss
            track = self.system[f'A{loc}'][0].get_absorption(
                thickness/nstep)*energy_map[:, n-1]
            
            # calculate remaining energy
            delta_e = self.system[f'A{loc}'][0].get_transmission(
                thickness/nstep)*energy_map[:, n-1]
            
            # update system
            self.system[f'A{loc}'][0].energies = np.clip(delta_e, 0, None)
            self.system[f'A{loc}'][0].generateCrossSections(
                                        self.system[f'A{loc}'][0].material, 
                                        self.system[f'A{loc}'][0].energies
                                        )

            # update arrays
            energy_map[:, n] = self.system[f'A{loc}'][0].energies
            track_map[:, n] = track

        # generate step positions for plotting
        zsteps = np.linspace(0, thickness, nstep)

        # reset material database:
        for n in range(len(cumulative_thickness)):
            self.system[f'A{loc}'][0].generateCrossSections(
                                    self.system[f'A{loc}'][0].material,
                                    hold_energies
                                    )

        return (track_map, energy_map, zsteps)
    
    def plot_array_dict(self, ax=None, yoff=0, seperator=True,colors=None):
        '''
        Plots dictionary generated by buildRail in a readable format. 
        Includes lengths and colours for distinct materials
        '''
        # assign colors
        prop_cycle = plt.rcParams['axes.prop_cycle']

        if colors is None:
            colors = prop_cycle.by_key()['color']
        
        material_order = []

        for key in self.system:
            s = self.system[key]
            material_order.append(s[0].materialstr)

        obs = {}
        legendentry = []
        for idx, o in enumerate(np.unique(material_order)):
            obs[o] = colors[idx]
            legendentry.append(Patch(facecolor=colors[idx], alpha=0.25))

        x = 0
        if ax == None:
            fig, ax = plt.subplots(1, 1)

        material_order = []
        length_order = []
        for s in self.system:
            key = self.system[s]
            material_order.append(key[0].materialstr)
            length_order.append(key[1])

        for idx, mat in enumerate(material_order):
            y = 0+yoff
            y2 = 1+yoff
            x2 = x+length_order[idx]
            # ax.add_patch(rect)

            xs = [x, x2]
            ys1 = [y, y]
            ys2 = [y2, y2]

            ax.fill_between(xs, ys1, ys2,
                            facecolor=obs[mat], alpha=0.25)
            # ax.plot([x2, x2], [y, y2], 'k--', lw=1)
            x = np.copy(x2)

        if seperator:
            ax.plot([0, x2], [y, y], 'k-')
        ax.set_xlabel('z (mm)')
        ax.legend(legendentry, obs, loc='center left',
                bbox_to_anchor=(1, 0.5))
    
