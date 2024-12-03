This document is for a general list of To-Dos, bug tracking etc. while LAMP is in havey development

# To Do

- Transform fluctuations? I think cv2 uses homography, not affine transform. But Skimage has prespective transform / warp, might work better? Need some testing? BUT some of this for ELI at least is because of the fibre bundle imaging
- Unit_convert(val,from,to)? Error out if not possible.
- plot_proc_shot can be general? if particular diagnostic info needed, get it (i.e. axis units)
- In fact, general get_proc_shot could be in diagnostic.py??
- Montage X axis? (not just shot string - although should have that as well)
- Do Montage pixel ROIs apply after resampling?? I don't think they do...
- unit_convert function? (val,from,to). E.g. m to mm, eV to J? Error if not compatiable.
- finish background subtraction functions/module in ImageProc
- MoU or disclaimer. How LAMP collaboration works.
- better Docs. Markdown good for now? Something better in future? https://about.readthedocs.com/?ref=readthedocs.org
- Read control file(database) and shot file(database). For Gemini, Have a look at Mirage SQL Lite file that it generates. Both of these are pandas dataframes that Matt uses and we might use going forward? APOLLO?
- XCOM for higher energy transmissions. Filter transmissions below 1 keV (fixed Al but none others)
- Once we are at a crude v1, make package and write at least an arXiv paper for accreditation? 
- Make screen montage function for all diagnostic output in single image? Like screenshot of live plotting on experiment. In experiment class? Diagnostics might need a function for it?

# Features / Thoughts...

- User defined diagnostics (in some local folder), that can extend core diagnostics. Say if you wanted to modify and have your own slightly different ESpec code.
- Same as above for DAQs and Metas? User defined versions that can extend. This would be important if LAMP was an installable package?
- LAMP script folder for things like, filter transmission plot, synchrotorn spectrum plot, or …?
- Metadata handling? (i.e. shot sheets). get_meta('meta_id', shot_dict)
- GitHub discussion page?

# Bugs

Please list any problems you encounter while using here (but let somebody know as well?)...

# Reminders

- Try to use the internal functionality of LAMP as much as possible. There are routines for image transformation, background correction, etc. In particular there are handy interface functions like run_img_calib()
- Use debug=False/True flags in functions to hide/show workings out and internals.
- Keep ROI selections until the final return funciton. I.e. get_*etc*. Otherwise conflicts?
