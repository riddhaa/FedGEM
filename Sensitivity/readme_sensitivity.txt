This directory contains all the code required to run the Sensitivity Study in the paper, and generate Figures 1 and 2.

This directory contains multiple subdirectories, whose name is of the format "<R_min>_<K>_<Experimental_Setting>", 
where R_min and K are as defined in the paper, and the experimental settings are the following:
	-nominal: Nominal setting
	-cluster: Cluster imbalance setting
	-client: Client imbalance setting

Each subdirectory contains 3 files: A .py file, a .m file, and a "rename_files.py" file. The steps to run the code are as follows.
	1) Run the .py file the same number of repetitions you would like to run the experiment. Each run creates a new .mat
	   file and stores it in the same subdirectory.
	2) Run the "rename_files.py" file to rename all the generated .mat files in sequential order.
	3) Run the .m file to aggregate the results of the experimental runs.

In order to generate Figure 1 in the paper, follow the following steps:
	1) Repeat the previous three steps for all the subdirectories in the Ablation directory. Make sure you do not delete any
	   variables from your MATLAB workspace as you run the .m files from all directories. Important variables containing
	   experimental results do not get overwritten, therefore all results are retained.
	2) Save the current MATLAB workspace in the Ablation directory with the name "collected_data_ar.mat".
	3) Run the plotting_ar.m file to generate the plots.

In order to generate Figure 2 in the paper, first change the any occurrence of "ar" in all the .m files in all subdirectories to 
"sil". Subsequently, repeat the previous three steps, replacing any instances of "ar" with "sil".

Note 1: In the .m files make sure to change the num_exp variable to reflect the number of repetitions you ran.
Note 2: In the code, "FedClem" or "clem" refer to our "FedGEM" algorithm.