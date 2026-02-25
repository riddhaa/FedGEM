num_exp = 50;

sil_vec_AFCL = zeros(1,num_exp);
sil_vec_k_FED = zeros(1,num_exp);
sil_vec_FedKmeans = zeros(1,num_exp);
sil_vec_FFCM1 = zeros(1,num_exp);
sil_vec_FFCM2 = zeros(1,num_exp);

sil_vec_GMM = zeros(1,num_exp);
sil_vec_DpGMM = zeros(1,num_exp);
sil_vec_kmeans = zeros(1,num_exp);

sil_vec_clem = zeros(1,num_exp);

for i = 1:num_exp
    filename = sprintf('FedClem_MNIST_%d.mat',i);
    load(filename);
    sil_vec_AFCL(i) = sil_AFCL;
    sil_vec_k_FED(i) = sil_k_FED;
    sil_vec_FedKmeans(i) = sil_FedKmeans;
    sil_vec_FFCM1(i) = sil_FFCM1;
    sil_vec_FFCM2(i) = sil_FFCM2;

    sil_vec_GMM(i) = sil_GMM;
    sil_vec_DpGMM(i) = sil_DpGMM;
    sil_vec_kmeans(i) = sil_kmeans;

    sil_vec_clem(i) = sil_clem;
end

ave_sil_AFCL = mean(sil_vec_AFCL);
ave_sil_k_FED = mean(sil_vec_k_FED);
ave_sil_FedKmeans = mean(sil_vec_FedKmeans);
ave_sil_FFCM1 = mean(sil_vec_FFCM1);
ave_sil_FFCM2 = mean(sil_vec_FFCM2);

ave_sil_GMM = mean(sil_vec_GMM);
ave_sil_DpGMM = mean(sil_vec_DpGMM);
ave_sil_kmeans = mean(sil_vec_kmeans);

ave_sil_clem = mean(sil_vec_clem);

std_sil_AFCL = std(sil_vec_AFCL);
std_sil_k_FED = std(sil_vec_k_FED);
std_sil_FedKmeans = std(sil_vec_FedKmeans);
std_sil_FFCM1 = std(sil_vec_FFCM1);
std_sil_FFCM2 = std(sil_vec_FFCM2);

std_sil_GMM = std(sil_vec_GMM);
std_sil_DpGMM = std(sil_vec_DpGMM);
std_sil_kmeans = std(sil_vec_kmeans);

std_sil_clem = std(sil_vec_clem);