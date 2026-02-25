num_exp = 50;

ar_vec_AFCL = zeros(1,num_exp);
ar_vec_k_FED = zeros(1,num_exp);
ar_vec_FedKmeans = zeros(1,num_exp);
ar_vec_FFCM1 = zeros(1,num_exp);
ar_vec_FFCM2 = zeros(1,num_exp);

ar_vec_GMM = zeros(1,num_exp);
ar_vec_DpGMM = zeros(1,num_exp);
ar_vec_kmeans = zeros(1,num_exp);

ar_vec_clem = zeros(1,num_exp);

for i = 1:num_exp
    filename = sprintf('FedClem_FrogA_%d.mat',i);
    load(filename);
    ar_vec_AFCL(i) = ar_AFCL;
    ar_vec_k_FED(i) = ar_k_FED;
    ar_vec_FedKmeans(i) = ar_FedKmeans;
    ar_vec_FFCM1(i) = ar_FFCM1;
    ar_vec_FFCM2(i) = ar_FFCM2;

    ar_vec_GMM(i) = ar_GMM;
    ar_vec_DpGMM(i) = ar_DpGMM;
    ar_vec_kmeans(i) = ar_kmeans;

    ar_vec_clem(i) = ar_clem;
end

ave_ar_AFCL = mean(ar_vec_AFCL);
ave_ar_k_FED = mean(ar_vec_k_FED);
ave_ar_FedKmeans = mean(ar_vec_FedKmeans);
ave_ar_FFCM1 = mean(ar_vec_FFCM1);
ave_ar_FFCM2 = mean(ar_vec_FFCM2);

ave_ar_GMM = mean(ar_vec_GMM);
ave_ar_DpGMM = mean(ar_vec_DpGMM);
ave_ar_kmeans = mean(ar_vec_kmeans);

ave_ar_clem = mean(ar_vec_clem);

std_ar_AFCL = std(ar_vec_AFCL);
std_ar_k_FED = std(ar_vec_k_FED);
std_ar_FedKmeans = std(ar_vec_FedKmeans);
std_ar_FFCM1 = std(ar_vec_FFCM1);
std_ar_FFCM2 = std(ar_vec_FFCM2);

std_ar_GMM = std(ar_vec_GMM);
std_ar_DpGMM = std(ar_vec_DpGMM);
std_ar_kmeans = std(ar_vec_kmeans);

std_ar_clem = std(ar_vec_clem);