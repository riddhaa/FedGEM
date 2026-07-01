ADAPTIVE SPLIT-REFINE-MERGE PREPROCESSING

All eight Table 6 benchmark programs now run the client-local preprocessing from
Algorithms 1 and 2 of the supplied thesis before constructing FedGEM. Each client
starts with K_g = 1. Candidate components are split along the leading eigenvector,
accepted by the responsibility-weighted BIC and minimum-mass tests, refined by
fixed-identity-covariance EM, and iteratively merged using the FedGEM overlap test.

The original random clusters-per-client vector is still used to generate the same
non-IID benchmark partitions and initialize the unchanged comparison methods. It is
not passed to FedGEM. The recovered per-client counts are saved in every result MAT
file as "recovered_cpc".

Defaults reproduce the values stated in Thesis Table 6.4:
  split perturbation delta: 1.0
  minimum child fraction tau_thresh: 0.05
  client overlap radius a_g: 1.0

Optional environment variables:
  FEDGEM_SPLIT_DELTA
  FEDGEM_SPLIT_TAU
  FEDGEM_OVERLAP_RADIUS
  FEDGEM_SPLIT_EM_STEPS
  FEDGEM_REFINE_EM_STEPS
  FEDGEM_SPLIT_MAX_COMPONENTS

Run each dataset exactly as described in readme_benchmarking.txt. The preprocessing
runs automatically and prints the recovered K_g for every client.
