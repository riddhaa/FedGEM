num_exp = 10;
len_vec = 4;
scal_mat_clem_clus = zeros(num_exp,len_vec);
scal_mat_AFCL_clus = zeros(num_exp,len_vec);
scal_mat_k_FED_clus = zeros(num_exp,len_vec);
scal_mat_FFCM1_clus = zeros(num_exp,len_vec);
scal_mat_FFCM2_clus = zeros(num_exp,len_vec);
scal_mat_FedKmeans_clus = zeros(num_exp,len_vec);
scal_mat_GMM_clus = zeros(num_exp,len_vec);
scal_mat_DpGMM_clus = zeros(num_exp,len_vec);

for i = 1:num_exp
    filename = sprintf('FedClem_Scal_clus_%d.mat',i);
    load(filename)
    scal_mat_clem_clus(i,:) = runtime_clem;
    scal_mat_AFCL_clus(i,:) = runtime_AFCL;
    scal_mat_k_FED_clus(i,:) = runtime_k_FED;
    scal_mat_FFCM1_clus(i,:) = runtime_FFCM1;
    scal_mat_FFCM2_clus(i,:) = runtime_FFCM2;
    scal_mat_FedKmeans_clus(i,:) = runtime_FedKmeans;
    scal_mat_GMM_clus(i,:) = runtime_GMM;
    scal_mat_DpGMM_clus(i,:) = runtime_DpGMM;
end

ave_scal_clus_clem = mean(scal_mat_clem_clus,1);
std_scal_clus_clem = std(scal_mat_clem_clus,[],1);

ave_scal_clus_AFCL = mean(scal_mat_AFCL_clus,1);
std_scal_clus_AFCL = std(scal_mat_AFCL_clus,[],1);

ave_scal_clus_FFCM1 = mean(scal_mat_FFCM1_clus,1);
std_scal_clus_FFCM1 = std(scal_mat_FFCM1_clus,[],1);

ave_scal_clus_FFCM2 = mean(scal_mat_FFCM2_clus,1);
std_scal_clus_FFCM2 = std(scal_mat_FFCM2_clus,[],1);

ave_scal_clus_FedKmeans = mean(scal_mat_FedKmeans_clus,1);
std_scal_clus_FedKmeans = std(scal_mat_FedKmeans_clus,[],1);

ave_scal_clus_GMM = mean(scal_mat_GMM_clus,1);
std_scal_clus_GMM = std(scal_mat_GMM_clus,[],1);

ave_scal_clus_DpGMM = mean(scal_mat_DpGMM_clus,1);
std_scal_clus_DpGMM = std(scal_mat_DpGMM_clus,[],1);

ave_scal_clus_k_FED = mean(scal_mat_k_FED_clus,1);
std_scal_clus_k_FED = std(scal_mat_k_FED_clus,[],1);
