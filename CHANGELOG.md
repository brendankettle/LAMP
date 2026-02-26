# Change Log
All notable changes to this project will be documented in this file.

## [0.2.0] - yyyy-mm-dd
 
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

## [0.1.0] - 2026-01-09
First major release.