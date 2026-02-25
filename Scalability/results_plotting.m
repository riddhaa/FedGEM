load collected_results.mat;
subplot(2,2,1)
semilogy(n_features_vec,ave_scal_feat_clem,'LineWidth',0.01)
hold on
errorbar(n_features_vec,ave_scal_feat_clem,std_scal_feat_clem,...
    'LineWidth',1)
errorbar(n_features_vec,ave_scal_feat_AFCL,std_scal_feat_AFCL,...
    'LineWidth',1)
errorbar(n_features_vec,ave_scal_feat_FFCM2,std_scal_feat_FFCM2,...
    'LineWidth',1)
errorbar(n_features_vec,ave_scal_feat_FedKmeans,std_scal_feat_FedKmeans,...
    'LineWidth',1)
legend({'','FedGEM','AFCL','FFCM-avg2','FedKmeans'},...
    'Interpreter','latex','Orientation','horizontal')
title('Features','Interpreter','latex')
xlabel('\#Features $d$','Interpreter','latex')
ylabel('Runtime [s]','Interpreter','latex')
grid on
axis([5 65 5e-3 1e4])
xticks([5 25 45 65])

subplot(2,2,2)
semilogy(n_sample_vec,ave_scal_samp_clem,'LineWidth',0.01)
hold on
errorbar(n_sample_vec,ave_scal_samp_clem,std_scal_samp_clem,...
    'LineWidth',1)
errorbar(n_sample_vec,ave_scal_samp_AFCL,std_scal_samp_AFCL,...
    'LineWidth',1)
errorbar(n_sample_vec,ave_scal_samp_FFCM2,std_scal_samp_FFCM2,...
    'LineWidth',1)
errorbar(n_sample_vec,ave_scal_samp_FedKmeans,std_scal_samp_FedKmeans,...
    'LineWidth',1)
title('Samples per Client','Interpreter','latex')
xlabel('\#Samples per Client $N_g$','Interpreter','latex')
ylabel('Runtime [s]','Interpreter','latex')
grid on
axis([500 6500 5e-3 1e4])
xticks([500 2500 4500 6500])

subplot(2,2,3)
semilogy(n_clusters_vec,ave_scal_clus_clem,'LineWidth',0.01)
hold on
errorbar(n_clusters_vec,ave_scal_clus_clem,std_scal_clus_clem,...
    'LineWidth',1)
errorbar(n_clusters_vec,ave_scal_clus_AFCL,std_scal_clus_AFCL,...
    'LineWidth',1)
errorbar(n_clusters_vec,ave_scal_clus_FFCM2,std_scal_clus_FFCM2,...
    'LineWidth',1)
errorbar(n_clusters_vec,ave_scal_clus_FedKmeans,std_scal_clus_FedKmeans,...
    'LineWidth',1)
title('Clusters','Interpreter','latex')
xlabel('\#Clusters $K$','Interpreter','latex')
ylabel('Runtime [s]','Interpreter','latex')
grid on
axis([5 65 5e-3 1e5])
xticks([5 25 45 65])

subplot(2,2,4)
semilogy(G_vec,ave_scal_cli_clem,'LineWidth',0.01)
hold on
errorbar(G_vec,ave_scal_cli_clem,std_scal_cli_clem,...
    'LineWidth',1)
errorbar(G_vec,ave_scal_cli_AFCL,std_scal_cli_AFCL,...
    'LineWidth',1)
errorbar(G_vec,ave_scal_cli_FFCM2,std_scal_cli_FFCM2,...
    'LineWidth',1)
errorbar(G_vec,ave_scal_cli_FedKmeans,std_scal_cli_FedKmeans,...
    'LineWidth',1)
title('Clients','Interpreter','latex')
xlabel('\#Clients $G$','Interpreter','latex')
ylabel('Runtime [s]','Interpreter','latex')
grid on
axis([5 65 5e-3 1e4])
xticks([5 25 45 65])
