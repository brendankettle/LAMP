# LAMP Documentation (in progress)
[Here is a Markdown cheat sheet]{https://www.markdownguide.org/cheat-sheet/}. 

## Aim/Ethos
The main goal of LAMP is to provide reusable diagnostic code that can be easily implemented at different laser facilities. 
This works by using an abstraction layer for accessing data, with a different DAQ module depending on the facility in question. 
Each diagnostic module will work through this system. 
Scripts to analysis data should be as simple as possible, reusable if possible, and rely on separate configuration/calibration files that are for the most part, human readable/writable.

## Quickstart

A few concepts

**Shot dictionaries** are how you define which data to analysis and how you interact with the DAQ. 
The format can be unique to the DAQ inquestions. 
Crucially, the diagnostics code should not depend on the specifics of the shot dictionaries, but rather always interact through the DAQ.
**Calibrations** are key to setting up diagnostics. 
~~A history file defines which calibrations apply when. An input file defines the required information for generating a calibration file.~~

## Requirements
toml
opencv?

### Starting a new experiment

1.	Decide on your folder structure. Examples are (will be) given in /templates/.
2.	Create a config file. The minimum required information is the DAQ name (see list of DAQs later) in a setup section, and the path to the data folder in paths section.

```
[setup]
DAQ = GeminiMirage
[paths]
data_folder = /Some/Path/To/All/Experiment/Data/
```

3.	Add a diagnostic to the diagnostic config file. The minimum required information is the name, the diagnostic module (type), and the data folder (contained in the root data folder).

```
[setup]
name = ESpecHigh
type = ESpec
data_folder = ./ESpec/
```

4.	Create a script. This will start by initiating the experiment using a root folder which points to LAMP, and passing an experiment config file.

## How LAMP works in general

Some text to describe the abstraction layer, how diagnostics are laoded etc.

Calibrations
Convention is to provide a calibration input file, that can be processed by the diagnostic, make_calib()?, that generates a calibration file for calling during other analysis.

## Diagnostics

### ESpec

Description of diagnostic workings. Including diagrams. Calibration input definitions

## Developing LAMP

General writing guidelines
- For functions in the DAQ / Diagnostic layers, try to keep named arguments? 
- Keep specific actions as separate subfunctions as to future proof the code or centralise changes in structure / formatting.

Writing a new diagnostic

General writing specifications
Should interface to the DAQ through the following functions:
- get_shot_data(diagnostic_name, shot_dictionary)
- get_shot_info?
- Calibration inputs should, where possible be written in JSON so that they are human readable?
- Calibration outputs should be saved as .pkl files where it is large data (arrays or images), or JSON files if it is single values?
- Results should be saved as .pkl files?

Writing a new DAQ interface
Required function calls
- get_shot_data(diagnostic_name, shot_dictionary)
- get_shot_info?

... Writing a new Meta interface

