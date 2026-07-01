STANDALONE (NON-FEDERATED) PREPROCESSING VALIDATION

Purpose
-------
This experiment treats all observations as one centralized dataset and evaluates the
adaptive split-refine-merge algorithm without running FedGEM. Labels are never given
to the clustering algorithm; they are used only for stratified splitting and metrics.

Method
------
1. Pool/load the complete dataset.
2. Make a reproducible stratified 70/30 train/test split.
3. Keep the original benchmark feature scale by default. Optionally fit a scaler on
   training features only with --standardize.
4. Run adaptive splitting from K=1 on the centralized training features.
5. Assign held-out observations using Gaussian posterior probabilities.
6. Report true and estimated K, count error, ARI, silhouette score, runtime, weights,
   and distances between estimated centroids and labeled class means.
7. Record the exact command/configuration in the metrics JSON for reproducibility.
8. Save centroids in the original feature scale as NPZ and metrics as JSON.

Examples
--------
From the Benchmarking directory:

  python standalone_validation.py --dataset Synthetic
  python standalone_validation.py --dataset MNIST
  python standalone_validation.py --dataset CIFAR-10
  python standalone_validation.py --dataset all

To run a feature-scale sensitivity check:

  python standalone_validation.py --dataset MNIST --standardize

To test another hyperparameter setting:

  python standalone_validation.py --dataset MNIST --delta 0.5 --tau 0.03 \
      --overlap-radius 0.8

Inputs
------
The runner uses the same files/loaders as the original benchmark programs:

  Synthetic: generated in memory as a quick smoke test
  MNIST: embeddings_vae_train.npy, embeddings_vae_test.npy,
         labels_train.npy, labels_test.npy
  FMNIST, EMNIST, CIFAR-10: x_data.npy, y_data.npy
  FrogA, FrogB: Frogs_MFCCs.csv
  Abalone, Waveform: downloaded through ucimlrepo

Outputs
-------
Files are written under Benchmarking/standalone_results by default:

  <dataset>_standalone_metrics.json
  <dataset>_standalone_centroids.npz

Important interpretation
------------------------
Correct recovery of K and useful centroids in this centralized experiment validates
the split-refine-merge algorithm independently. A separate federated experiment is
still required to establish that FedGEM successfully extends or transports the method
to decentralized, non-IID client data.
