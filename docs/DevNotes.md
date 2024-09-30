This document is for a general list of To-Dos, bug tracking etc. while LAMP is in havey development

# To Do

- Make sure filenames and lists work instead of shot dictionaries for shot loading. I.e usual focal spot images saved out of DAQ for example
- data_type for diagnostics could be set in diagnostic class?
- plot_proc_shot can be general? if particular diagnostic info needed, get it (i.e. axis units)
- In fact, general get_proc_shot could be in diagnostic.py??
- unit_convert function? (val,from,to). E.g. m to mm, eV to J? Error if not compatiable.
- finish background subtraction functions/module in ImageProc
- MoU or disclaimer. How LAMP collaboration works.
- better Docs. Markdown good for now? Something better in future? https://about.readthedocs.com/?ref=readthedocs.org
-  Read control file(database) and shot file(database). For Gemini, Have a look at Mirage SQL Lite file that it generates. Both of these are pandas dataframes that Matt uses and we might use going forward? APOLLO?
- Fix up window / mac paths. Is this kind of done using Path()? Need to make sure it's used everywhere...
- XCOM for higher energy transmissions. Filter transmissions below 1 keV (fixed Al but none others)
- Once we are at a crude v1, make package and write at least an arXiv paper for accreditation? 
- Plotting Module could be much better...
- Make screen montage function for all diagnostic output in single image? Like screenshot of live plotting on experiment. In experiment class? Diagnostics might need a function for it?
- Montage plotting needs some work for ROIs etc (and should be in plotting class?)

# Features / Thoughts...

- LAMP script folder for things like, filter transmission plot, synchrotorn spectrum plot, or …?
- Metadata handling? (i.e. shot sheets). get_meta('meta_id', shot_dict)
- GitHub discussion page?

# Bugs

Please list any problems you encounter while using here (but let somebody know as well?)...
