num_exp = 50;

runtime_vec_AFCL = zeros(1,num_exp);
runtime_vec_k_FED = zeros(1,num_exp);
runtime_vec_FedKmeans = zeros(1,num_exp);
runtime_vec_FFCM1 = zeros(1,num_exp);
runtime_vec_FFCM2 = zeros(1,num_exp);

runtime_vec_GMM = zeros(1,num_exp);
runtime_vec_DpGMM = zeros(1,num_exp);

runtime_vec_clem = zeros(1,num_exp);

for i = 1:num_exp
    filename = sprintf('FedClem_EMNIST_%d.mat',i);
    load(filename);
    runtime_vec_AFCL(i) = runtime_AFCL;
    runtime_vec_k_FED(i) = runtime_k_FED;
    runtime_vec_FedKmeans(i) = runtime_FedKmeans;
    runtime_vec_FFCM1(i) = runtime_FFCM1;
    runtime_vec_FFCM2(i) = runtime_FFCM2;

    runtime_vec_GMM(i) = runtime_GMM;
    runtime_vec_DpGMM(i) = runtime_DpGMM;

    runtime_vec_clem(i) = runtime_clem;
end

ave_runtime_AFCL = mean(runtime_vec_AFCL);
ave_runtime_k_FED = mean(runtime_vec_k_FED);
ave_runtime_FedKmeans = mean(runtime_vec_FedKmeans);
ave_runtime_FFCM1 = mean(runtime_vec_FFCM1);
ave_runtime_FFCM2 = mean(runtime_vec_FFCM2);

ave_runtime_GMM = mean(runtime_vec_GMM);
ave_runtime_DpGMM = mean(runtime_vec_DpGMM);

ave_runtime_clem = mean(runtime_vec_clem);

std_runtime_AFCL = std(runtime_vec_AFCL);
std_runtime_k_FED = std(runtime_vec_k_FED);
std_runtime_FedKmeans = std(runtime_vec_FedKmeans);
std_runtime_FFCM1 = std(runtime_vec_FFCM1);
std_runtime_FFCM2 = std(runtime_vec_FFCM2);

std_runtime_GMM = std(runtime_vec_GMM);
std_runtime_DpGMM = std(runtime_vec_DpGMM);

std_runtime_clem = std(runtime_vec_clem);