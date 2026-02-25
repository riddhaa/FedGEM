num_exp = 10;
len_vec = 4;
scal_mat_clem_feat = zeros(num_exp,len_vec);
scal_mat_AFCL_feat = zeros(num_exp,len_vec);
scal_mat_k_FED_feat = zeros(num_exp,len_vec);
scal_mat_FFCM1_feat = zeros(num_exp,len_vec);
scal_mat_FFCM2_feat = zeros(num_exp,len_vec);
scal_mat_FedKmeans_feat = zeros(num_exp,len_vec);
scal_mat_GMM_feat = zeros(num_exp,len_vec);
scal_mat_DpGMM_feat = zeros(num_exp,len_vec);

for i = 1:num_exp
    filename = sprintf('FedClem_Scal_feat_%d.mat',i);
    load(filename)
    scal_mat_clem_feat(i,:) = runtime_clem;
    scal_mat_AFCL_feat(i,:) = runtime_AFCL;
    scal_mat_k_FED_feat(i,:) = runtime_k_FED;
    scal_mat_FFCM1_feat(i,:) = runtime_FFCM1;
    scal_mat_FFCM2_feat(i,:) = runtime_FFCM2;
    scal_mat_FedKmeans_feat(i,:) = runtime_FedKmeans;
    scal_mat_GMM_feat(i,:) = runtime_GMM;
    scal_mat_DpGMM_feat(i,:) = runtime_DpGMM;
end

ave_scal_feat_clem = mean(scal_mat_clem_feat,1);
std_scal_feat_clem = std(scal_mat_clem_feat,[],1);

ave_scal_feat_AFCL = mean(scal_mat_AFCL_feat,1);
std_scal_feat_AFCL = std(scal_mat_AFCL_feat,[],1);

ave_scal_feat_FFCM1 = mean(scal_mat_FFCM1_feat,1);
std_scal_feat_FFCM1 = std(scal_mat_FFCM1_feat,[],1);

ave_scal_feat_FFCM2 = mean(scal_mat_FFCM2_feat,1);
std_scal_feat_FFCM2 = std(scal_mat_FFCM2_feat,[],1);

ave_scal_feat_FedKmeans = mean(scal_mat_FedKmeans_feat,1);
std_scal_feat_FedKmeans = std(scal_mat_FedKmeans_feat,[],1);

ave_scal_feat_GMM = mean(scal_mat_GMM_feat,1);
std_scal_feat_GMM = std(scal_mat_GMM_feat,[],1);

ave_scal_feat_DpGMM = mean(scal_mat_DpGMM_feat,1);
std_scal_feat_DpGMM = std(scal_mat_DpGMM_feat,[],1);

ave_scal_feat_k_FED = mean(scal_mat_k_FED_feat,1);
std_scal_feat_k_FED = std(scal_mat_k_FED_feat,[],1);
