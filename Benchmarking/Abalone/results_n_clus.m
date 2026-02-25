num_exp = 50;

n_clusters_vec_AFCL = zeros(1,num_exp);
n_clusters_vec_DpGMM = zeros(1,num_exp);
n_clusters_vec_clem = zeros(1,num_exp);

for i = 1:num_exp
    filename = sprintf('FedClem_Abalone_%d.mat',i);
    load(filename);
    n_clusters_vec_AFCL(i) = n_clusters_AFCL;
    n_clusters_vec_DpGMM(i) = n_clusters_DpGMM;
    n_clusters_vec_clem(i) = n_clusters_clem;
end

ave_n_clusters_AFCL = mean(n_clusters_vec_AFCL);
ave_n_clusters_DpGMM = mean(n_clusters_vec_DpGMM);
ave_n_clusters_clem = mean(n_clusters_vec_clem);

std_n_clusters_AFCL = std(n_clusters_vec_AFCL);
std_n_clusters_DpGMM = std(n_clusters_vec_DpGMM);
std_n_clusters_clem = std(n_clusters_vec_clem);