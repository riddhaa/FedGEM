num_exp = 10;
len_vec = 4;
scal_mat_clem_samp = zeros(num_exp,len_vec);
scal_mat_AFCL_samp = zeros(num_exp,len_vec);
scal_mat_k_FED_samp = zeros(num_exp,len_vec);
scal_mat_FFCM1_samp = zeros(num_exp,len_vec);
scal_mat_FFCM2_samp = zeros(num_exp,len_vec);
scal_mat_FedKmeans_samp = zeros(num_exp,len_vec);
scal_mat_GMM_samp = zeros(num_exp,len_vec);
scal_mat_DpGMM_samp = zeros(num_exp,len_vec);

for i = 1:num_exp
    filename = sprintf('FedClem_Scal_samp_%d.mat',i);
    load(filename)
    scal_mat_clem_samp(i,:) = runtime_clem;
    scal_mat_AFCL_samp(i,:) = runtime_AFCL;
    scal_mat_k_FED_samp(i,:) = runtime_k_FED;
    scal_mat_FFCM1_samp(i,:) = runtime_FFCM1;
    scal_mat_FFCM2_samp(i,:) = runtime_FFCM2;
    scal_mat_FedKmeans_samp(i,:) = runtime_FedKmeans;
    scal_mat_GMM_samp(i,:) = runtime_GMM;
    scal_mat_DpGMM_samp(i,:) = runtime_DpGMM;
end

ave_scal_samp_clem = mean(scal_mat_clem_samp,1);
std_scal_samp_clem = std(scal_mat_clem_samp,[],1);

ave_scal_samp_AFCL = mean(scal_mat_AFCL_samp,1);
std_scal_samp_AFCL = std(scal_mat_AFCL_samp,[],1);

ave_scal_samp_FFCM1 = mean(scal_mat_FFCM1_samp,1);
std_scal_samp_FFCM1 = std(scal_mat_FFCM1_samp,[],1);

ave_scal_samp_FFCM2 = mean(scal_mat_FFCM2_samp,1);
std_scal_samp_FFCM2 = std(scal_mat_FFCM2_samp,[],1);

ave_scal_samp_FedKmeans = mean(scal_mat_FedKmeans_samp,1);
std_scal_samp_FedKmeans = std(scal_mat_FedKmeans_samp,[],1);

ave_scal_samp_GMM = mean(scal_mat_GMM_samp,1);
std_scal_samp_GMM = std(scal_mat_GMM_samp,[],1);

ave_scal_samp_DpGMM = mean(scal_mat_DpGMM_samp,1);
std_scal_samp_DpGMM = std(scal_mat_DpGMM_samp,[],1);

ave_scal_samp_k_FED = mean(scal_mat_k_FED_samp,1);
std_scal_samp_k_FED = std(scal_mat_k_FED_samp,[],1);
