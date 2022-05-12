Tools for converting Super-K data files into the WatChMaL h5 data format.

There are two example python scripts here. Both take one input file for signal and one input file for background and generate the output h5 file with both the signal and background events. The scripts also generate a train-val-test indices split file, which must be supplied to WatChMaL along with the data file. This split file is analogous to the file generated in [DataTools/root_utils/Create_indices_file.ipynb](../root_utils/Create_indices_file.ipynb) for IWCD mPMT data.

The [preprocess_wit.py](preprocess_wit.py) script takes files in the WIT data format as input for both the signal and background.

The [preprocess_skroot_wit.py](preprocess_skroot_wit.py) script takes a background file in the WIT format and a signal file in the standard SKRoot format generated with the Super-K Fortran libraries. However, these files are not readable in python with either uproot or PyRoot. The relevant root branches must first be copied into a new file with the [copy_branches](copy_branches/copy_branches.cc) program.
