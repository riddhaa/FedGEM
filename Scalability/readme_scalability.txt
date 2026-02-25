This directory contains the code required to run the Scalability Study described in Appendix E.2. Note that the Runtime study in Appendix E.1 can be run using the same code provided in the Benchmarking Study. 

This directory contains 4 subdirectories, described next:
	-clients: increasing number of clients
	-clusters: increasing number of clusters
	-features: increasing number of features
	-samples: increasing number of training samples

Each subdirectory contains a .py file, a .m file, and a "rename_files.py" file. In order to recreate Figure 3, follow the following
steps:
	1) Navigate to each subdirectory.
	2) Run the .py file the same number of repetitions you would like to run the experiment. Each run creates a new .mat
	   file and stores it in the same subdirectory.
	3) Run the "rename_files.py" file to rename the .mat files in sequential order.
	4) Run the .m file, and make sure not to clear the workspace.
	5) Repeat steps 1-4 with all 4 subdirectories.
	6) Save the MATLAB workspace in the Scalability subdirectory under the name "collected_results.mat".
	7) Run the "results_plotting.m" file to generate Figure 3.

Note 1: In the .m files make sure to change the num_exp variable to reflect the number of repetitions you ran.
Note 2: In the code, "FedClem" or "clem" refer to our "FedGEM" algorithm.