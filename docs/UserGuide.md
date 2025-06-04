<img src="./imgs/LAMPLogo.png" alt="LAMP" width="300"/>

<!--- https://www.markdownguide.org/ --->

**L**aser-plasma **A**nalysis for **M**ultiple **P**latforms

# Documentation

## Aim / Ethos
The main goal of LAMP is to provide a **framework for processing data from laser-plasma experiments** at different laser facilities. 
It enables access to reusable diagnostic code and scripts that can be quickly implemented (with a bit of know how). 
This works by using an **abstraction layer for accessing data**, with a different layer (known as a **DAQ module**) depending on the facility in question. 
All analysis code works through **diagnostic modules**, which in turn use this DAQ system, but also use **shared libaries and utlities for common processing steps** like background subtraction, spatial transforms, etc.
The unique properties of any diagnostics on any given experiment are set using **calibration input decks**, that are for the most part human readable/writable.
With the above in place, **short, simple and reusable scripts** are the entry way for analysing experimental data, calling on LAMP functionality where needs be.
This should **reduce the rewriting of code and streamline analysis**, allowing online experiment analysis, but also hopefully reduce workload, identify bugs, and make the working of diagnostics accessible.

It is not a means of taking data (such as a facility DAQ system) or a simulation tool.

## Using LAMP
Need to be clear, by using LAMP, you are agreeing to the collaboration terms.
Don’t put code into it, if you don’t want it shared with others.
LAMP has no responsibility for analysis outcomes.
It is “Open-access” rather than open source; requests for access will be accepeted if the user agrees to the memoranium of understanding.

## Table of contents
1. [Getting Started](#getting-started)
2. [Developing LAMP](#developing-lamp)
3. [Diagnostic Manuals](#diagnostic-manuals)
4. [DAQs](#DAQs)

<a name="getting-started"></a>
#1. Getting Started

## Installation 

As LAMP is still currently under heavy development, it has not been packaged and so "installation" is currently a bit rough and ready.

My (BK) recommendation currently is to follow these steps:

- Download something like [GitHub desktop](https://desktop.github.com/download/) to manage the different experiment and LAMP repositories on your computer, and to push/pull updates. If you don't know how to use git, I would recommend googling some tutorials.
- Download a local copy of the experiment repository.
- Copy *_local.toml* to *local.toml* and edit the data folder path to point to your local copy.
- Download a local copy of the LAMP repository to INSIDE this experiment folder. If you are working on multiple experiments, it helps to create an alias for each LAMP copy.
- The figure below depicts an example folder layout for an experiment repository, with LAMP inside it.
- **Make sure you push/pull all updates before and after working on the experiments or LAMP**.

Note that currently I am not expecting anybody to setup an experiment repository, I (BK) can do that. Please ask if you need help with this.

![An Example file structure for an experiment repository](./imgs/ExampleExpRepo.png)

Requirements (at least):

- numpy
- matplotlib
- scipy
- skimage
- toml
- opencv
- pandas

## An example script 

```
# -----------------------------------------------------------------------------
# import LAMP - this will be much easier when installable as a package
import sys, os
ROOT_FOLDER = os.path.dirname(__file__) + '/../../' # point this to the folder containing LAMP and/or the config files
sys.path.append(ROOT_FOLDER)
from LAMP import Experiment
# -----------------------------------------------------------------------------
import matplotlib.pyplot as plt 

# create experiment object
ex = Experiment(ROOT_FOLDER)

# get ESpec diagnostic object
eSpec = ex.get_diagnostic('eSpec')

# define a shot
shot_dict = {'date': 20250218, 'run': 'run05', 'shotnum': 40}

# plot
fig, ax = eSpec.plot_proc_shot(shot_dict, debug=True)
plt.tight_layout()
plt.show(block=True)
```

1. **Import LAMP** - this is a bit of a fudge at the minute, until we package up LAMP. 
2. **Setup the experiment object** by passing a path to the root anylsis folder (experiment repository clone). This reads the configuration files and initiates the framework.
3. **Get a diagnostic object.** This could also be a results object or meta information object. Each of these objects will have their own functionality, but interact with experiment data or processed information through LAMP subroutines, or the interfaces defined in the different DAQ modules.
4. **Define a shot dictionary**; these are key to how LAMP accesses data universally. It is how you point the code to specific shot data. It will also dictate the specific calibration used by the diagnostic.
5. **Use an analysis function**. Here we are plotting a processed shot, with some debug info. Rach diagnostic will have a range of functions. See the user guide or example scripts for each.


## Configuring diagnostics and experiment details:

Most of the configuration or calibration inputs file make use of **TOML (Tom's Obvious Minimal Language)**. I would recommend having a quick look at the website if you are unfamiliar - [https://toml.io/en/](https://toml.io/en/) - it is fairly straightforward. 

### **global.toml** 
This file contains experiment settings common for everybody collaborating on the analysis (as opposed to local.toml, which contains user specific details, such as local file paths). 
Currently, the global file usually just specifys which DAQ module to use, and any extra settings. 
Also some relative filepaths, for things like the root calibrations folder.

Again, most people should not need to edit this file, it is mostly for setting up the experiment.

### **diagnostics.toml** 
This file contains a list of each of the diagnostics to load for the experiment. 
Each section should be labelled with the diagnostic name, and contain a set list of parameters. 
Edit this if you want to add a new diagnostic to the experiment. 
An example is given below. 

```
[eSpecHigh]
type = 'ESpec'
data_folder = '/eSpec/' 
data_stem = 'ESpec_' # empty for some DAQs 
data_type = 'image'
data_ext = 'tif'
calib_subfolder = './eSpecs/' # optional
calib_file = 'eSpec_calibs.toml'

[Betatron]
type = 'XrayFilterArray'
data_folder = '/Betatron/' 
data_stem = 'OnAxis_' # empty for some DAQs 
data_type = 'image'
data_ext = 'tif'
calib_subfolder = './Betatron/' # optional
calib_file = 'Betatron_calibs.toml'
```

### **Calibrations** 
The calibration files you define are key to setting up a diagnostic with the specific details for the experiment (and which can change occasionally throughout said experiment). 
They are like an input deck for the diagnostic.
Things like geometry distances, filter thicknesses, etc.
Each diagnostic will have one calibration file, containing a list of seperate cases where the configuration has changed. 
Each configuration case will have its own named block with a unique ID, containing seperate calibration details.
It will also define the start and end shot dictionaries, indicating when the specific calibration should be applied. 
Think of it as a history list of when the diagnostic is altered throughout an experiment, each block being a specific setup.
See the example below.

```
[some_unique_calibration_id]
	notes = 'The example start to a calibraiton file'
	proc_file = 'OutPutFileName.pkl' # optional; if making processed calibrations
	start = {date = 20240101, run = 'run01', shot = 0}
	end = {date = 20240131, run = 'run99', shot = 0}
	
	[some_subset_of_settings]
	pixel_X = 100
	pixel_Y = 200
	pixel_Z = 300
```

Some diagnostics will process calibration inputs into a processed binary file (the filepath given by *proc\_file*).
For example the transformation details for warping a screen, or the curve relating detector pixel to spectral value.
Configuration details that aren't human readable or contain alot of information.
These are usually generated by a make_calib() function.

Within each seperate calibration ID, you can define sub-categories of the configuration, that do not have to be unique.


## Contributing
When you have made a (hopefully somewhat stable) update to your diagnostic code, push the changes.
For experiment scripts they will be part of an experiment repository update.
For changes to the core workings of LAMP (i.e. diagnostic code), you will have to update the LAMP repository.
For now, BK will have to accept these changes and merge them to the master version, but you can keep on developing on your side. (Not sure if this is still true?)
If multiple people are developing the same diagnostic code, we might want to have different modules, or co-ordinate the writing.
One of the least invasive or safe ways to add to a diagnostic is to add a your own new subroutine or function, that builds on the existing code, rather than modify it. 
Please try to follow the structures/conventions in place with existing LAMP code. 
For more details see the development section.



## General overview of how LAMP works

Below is my best effort of a pictorial description of how LAMP actually works! Or rather **will** work.

![Flowchat showing how LAMP works](./imgs/HowLAMPWorks.png)
	

## Starting a new experiment

NOTE: This should already be done within an experiment repository, most people won't have to do this. 
If you need help, ask BK, but normally I have set all this up.
I will eventually put some help here...

<a name="developing-lamp"></a> 
#2. Developing LAMP 

## General guidelines
- For functions in the DAQ / Diagnostic layers, try to keep named arguments? 
- Keep specific actions as separate subfunctions as to future proof the code or centralise changes in structure / formatting.

For writing diagnostics, alongside data access, LAMP has routines for;
Image transformation
Background correction
ROI definitions
Montage plotting
Etc.


## Writing a new diagnostic

I need to finish this section...

Where it makes sense, have functions following this nonclemature:

- `get_proc_shot(shot_dict)`
- `make_calib()`
- ...

Make use of common / shared functions such as:

- `run_img_calib()` ... more info on this...
- ...

Calibration statements that are autoprocessed by the above function:

+table+

Should interface to the DAQ through the following functions:

- `get_shot_data(diagnostic_name, shot_dictionary)`
- `get_shot_info?`
- Use calibration framework
- Calibration outputs should be saved as .pkl files where it is large data (arrays or images)

Debug flags; Important for showing inner workings of processing. All functions should have an optional parameter to turn on plotting / output for tracking what has been done, finding problems.


## Writing a new DAQ interface

To Do. Required function calls?

- `get_shot_data(diagnostic_name, shot_dictionary)`
- `get_shot_info?`
- `timeline?`


<hr/>

<a name="diagnostic-manuals"></a> 
#3. Diagnostic Manuals

The following portion of this documentation will focus on the details of using each diagnostic.
Please keep the following in mind.
Student learning - realistically you still won’t be able to use the code without knowing how the diagnostic works; LAMP’s goal is to remove the bulk of the coding work for analysis. Good documentation will aide in understanding & learning. If somebody has a primary goal of developing a diagnostic then they can dig deeper.


## ESpec

### Description
Including diagrams

### References

- General
	-  
- Charge calibration
	- G. Boutoux et al. Rev. Sci. Instrum. 86, 113304 (2015) [https://doi.org/10.1063/1.4936141](https://doi.org/10.1063/1.4936141)
	- Maddox?
- Advanced


### Key Functions

Along with internal functions (that can be used if needed), here is a list of common user callable functions:

`get_proc_shot(shot_dict, ...) / plot_proc_shot(shot_dict, ...)`

**Returns:** *img, x, y* 

Where...

`get_spectrum() / get_spectra() / plot_spectrum()`

`get_spectrum_metrics() and get_spectra_metrics()`

`get_charge()`

`montage()`

### Calibration inputs

dispersion
divergence


## XAS

...

<a name="DAQs"></a>
#4. DAQs

## Gemini Mirage

Requirements:

- sqlite3

