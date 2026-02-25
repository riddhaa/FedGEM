num_exp = 10;
len_vec = 4;
scal_mat_clem_cli = zeros(num_exp,len_vec);
scal_mat_AFCL_cli = zeros(num_exp,len_vec);
scal_mat_k_FED_cli = zeros(num_exp,len_vec);
scal_mat_FFCM1_cli = zeros(num_exp,len_vec);
scal_mat_FFCM2_cli = zeros(num_exp,len_vec);
scal_mat_FedKmeans_cli = zeros(num_exp,len_vec);
scal_mat_GMM_cli = zeros(num_exp,len_vec);
scal_mat_DpGMM_cli = zeros(num_exp,len_vec);

for i = 1:num_exp
    filename = sprintf('FedClem_Scal_cli_%d.mat',i);
    load(filename)
    scal_mat_clem_cli(i,:) = runtime_clem;
    scal_mat_AFCL_cli(i,:) = runtime_AFCL;
    scal_mat_k_FED_cli(i,:) = runtime_k_FED;
    scal_mat_FFCM1_cli(i,:) = runtime_FFCM1;
    scal_mat_FFCM2_cli(i,:) = runtime_FFCM2;
    scal_mat_FedKmeans_cli(i,:) = runtime_FedKmeans;
    scal_mat_GMM_cli(i,:) = runtime_GMM;
    scal_mat_DpGMM_cli(i,:) = runtime_DpGMM;
end

ave_scal_cli_clem = mean(scal_mat_clem_cli,1);
std_scal_cli_clem = std(scal_mat_clem_cli,[],1);

ave_scal_cli_AFCL = mean(scal_mat_AFCL_cli,1);
std_scal_cli_AFCL = std(scal_mat_AFCL_cli,[],1);

ave_scal_cli_FFCM1 = mean(scal_mat_FFCM1_cli,1);
std_scal_cli_FFCM1 = std(scal_mat_FFCM1_cli,[],1);

ave_scal_cli_FFCM2 = mean(scal_mat_FFCM2_cli,1);
std_scal_cli_FFCM2 = std(scal_mat_FFCM2_cli,[],1);

ave_scal_cli_FedKmeans = mean(scal_mat_FedKmeans_cli,1);
std_scal_cli_FedKmeans = std(scal_mat_FedKmeans_cli,[],1);

ave_scal_cli_GMM = mean(scal_mat_GMM_cli,1);
std_scal_cli_GMM = std(scal_mat_GMM_cli,[],1);

ave_scal_cli_DpGMM = mean(scal_mat_DpGMM_cli,1);
std_scal_cli_DpGMM = std(scal_mat_DpGMM_cli,[],1);

ave_scal_cli_k_FED = mean(scal_mat_k_FED_cli,1);
std_scal_cli_k_FED = std(scal_mat_k_FED_cli,[],1);
