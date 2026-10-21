# Change Log
All notable changes to this project will be documented in this file.

## [0.3.0] - 2026-08-??

...

### Added
- Begun implementation of standalone single file operation. Note that calibration files for diagnostics still need defined in a seperate file. Need to work on in-file definitions and best way to handle this.

### Fixed
- 

## [0.2.1] - 2026-06-16

Some some bug fixes and feature additions

### Added
- GeminiMirage DAQ now supports filename_format modifiers. Should port this across to other DAQs soon, or maybe the core approach could be something like this.

### Fixed
- Default to text file type in load_file() if type (which actually uses extension) is not matched. This is a little confusing and not great - a full IO subsystem would be much better...
- Fixed montage issues for num_rows and transposing

## [0.2.0] - 2026-03-03
 
Bug fixes and feature additions.
 
### Added
- Support for text based diagnostics (work still to be done).
- Fitting and stats functionality for Focal Spot diagnostic

### Changed
- GeminiDAQ: Timeframes can be list of shots (which get ignored in subfunctions). Note; need to Update Apollo functionality (at least).
- Some minor changes to how calibrations are loaded.
 
### Fixed
- Some Path() instance checks (not strings)
- Diagnostic variables defined in init() rather than as class attributes, so they don't change across different diagnostics instances (python is whacky sometimes!)
- Adding root folder to system path in Experiment() startup; this fixes user DAQ problems.

## [0.1.0] - 2026-01-09
First major release.