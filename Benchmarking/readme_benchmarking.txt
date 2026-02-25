This directory contains all the code required to generate Tables 1,2,6, and 7 from the paper.

The directory contains 8 subdirectories, one per dataset tested.

Each subdirectory contains a main .py file, a "rename_files.py" file, and 3 .m files (or 4 .m files for the image datasets). The results can be obtained for each dataset as follows.
	1) Navigate to the subdirectory associated with the dataset of interest.
	2) For the Frog A, Frog B, MNIST, and CIFAR-10 datasets download the data from the sources cited in the paper. Data for Frog A and Frog B should be in .csv format, whereas data for MNIST and CIFAR-10 should be in .npy format.
	3) For the FMNIST, EMNIST, and CIFAR-10 datasets, run the 'feature_extraction.py' file.
	2) Run the main .py file the same number of repetitions you would like to run the experiment. Each run creates a new .mat file and stores it in the same subdirectory.
	3) Run the "rename_files.py" file to rename the generated .mat files in sequential order.
	4) Run the "results_ar.m", "results_sil.m", or "results_n_clus.m" file to output the Adjusted Rand Index, Silhouette Score, and number of clusters results, respectively. Additionally, you can run the "results_runtime.m" where available to extract the runtime results presented in Table 7. The variables starting with "ave" indicate the mean over the repetitions, whereas ones starting with "std" indicate the standard deviation.

Note 1: In the .m files make sure to change the num_exp variable to reflect the number of repetitions you ran.
Note 2: In the code, "FedClem" or "clem" refer to our "FedGEM" algorithm.
Note 3: You must download the file "Frogs_MFCCs.csv" from the reference to the Anuran Calls dataset, and add it to the FrogA and FrogB subdirectories before running the code therein.