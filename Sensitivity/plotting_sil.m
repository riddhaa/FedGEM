load 'collected_data_sil.mat'

acc_clem_nom = [acc_clem_1_10_nom, acc_clem_2_10_nom, acc_clem_4_10_nom, acc_clem_6_10_nom, acc_clem_8_10_nom];
acc_clem_gro = [acc_clem_1_10_gro, acc_clem_2_10_gro, acc_clem_4_10_gro, acc_clem_6_10_gro, acc_clem_8_10_gro];
acc_clem_cla = [acc_clem_1_10_cla, acc_clem_2_10_cla, acc_clem_4_10_cla, acc_clem_6_10_cla, acc_clem_8_10_cla];

acc_central_nom = [acc_central_1_10_nom, acc_central_2_10_nom, acc_central_4_10_nom, acc_central_6_10_nom, acc_central_8_10_nom];
acc_central_gro = [acc_central_1_10_gro, acc_central_2_10_gro, acc_central_4_10_gro, acc_central_6_10_gro, acc_central_8_10_gro];
acc_central_cla = [acc_central_1_10_cla, acc_central_2_10_cla, acc_central_4_10_cla, acc_central_6_10_cla, acc_central_8_10_cla];

std_clem_nom = [std_clem_1_10_nom, std_clem_2_10_nom, std_clem_4_10_nom, std_clem_6_10_nom, std_clem_8_10_nom];
std_clem_gro = [std_clem_1_10_gro, std_clem_2_10_gro, std_clem_4_10_gro, std_clem_6_10_gro, std_clem_8_10_gro];
std_clem_cla = [std_clem_1_10_cla, std_clem_2_10_cla, std_clem_4_10_cla, std_clem_6_10_cla, std_clem_8_10_cla];

std_central_nom = [std_central_1_10_nom, std_central_2_10_nom, std_central_4_10_nom, std_central_6_10_nom, std_central_8_10_nom];
std_central_gro = [std_central_1_10_gro, std_central_2_10_gro, std_central_4_10_gro, std_central_6_10_gro, std_central_8_10_gro];
std_central_cla = [std_central_1_10_cla, std_central_2_10_cla, std_central_4_10_cla, std_central_6_10_cla, std_central_8_10_cla];

trunc_std_clem_nom = truncate_std(acc_clem_nom,std_clem_nom);
trunc_std_clem_gro = truncate_std(acc_clem_gro,std_clem_gro);
trunc_std_clem_cla = truncate_std(acc_clem_nom,std_clem_cla);

trunc_std_central_nom = truncate_std(acc_central_nom,std_central_nom);
trunc_std_central_gro = truncate_std(acc_central_gro,std_central_gro);
trunc_std_central_cla = truncate_std(acc_central_nom,std_central_cla);

ave_n_cl_nom = [ave_n_cl_1_10_nom, ave_n_cl_2_10_nom, ave_n_cl_4_10_nom, ave_n_cl_6_10_nom, ave_n_cl_8_10_nom];
ave_n_cl_gro = [ave_n_cl_1_10_gro, ave_n_cl_2_10_gro, ave_n_cl_4_10_gro, ave_n_cl_6_10_gro, ave_n_cl_8_10_gro];
ave_n_cl_cla = [ave_n_cl_1_10_cla, ave_n_cl_2_10_cla, ave_n_cl_4_10_cla, ave_n_cl_6_10_cla, ave_n_cl_8_10_cla];

std_n_cl_nom = [std_n_cl_1_10_nom, std_n_cl_2_10_nom, std_n_cl_4_10_nom, std_n_cl_6_10_nom, std_n_cl_8_10_nom];
std_n_cl_gro = [std_n_cl_1_10_gro, std_n_cl_2_10_gro, std_n_cl_4_10_gro, std_n_cl_6_10_gro, std_n_cl_8_10_gro];
std_n_cl_cla = [std_n_cl_1_10_cla, std_n_cl_2_10_cla, std_n_cl_4_10_cla, std_n_cl_6_10_cla, std_n_cl_8_10_cla];


R_min_vec = [1,2,4,6,8];
true_cl_x = linspace(0,9,100);
true_cl_y = ones(1,100)*10;

subplot(2,3,1)
errorbar(R_min_vec, acc_central_nom, std_central_nom, trunc_std_central_nom,...
    'Color',[0.3010 0.7450 0.9330],'LineWidth',1)
hold on
errorbar(R_min_vec, acc_clem_nom, std_clem_nom, trunc_std_clem_nom,...
    'Color',[0.4940 0.1840 0.5560],'LineWidth',1)
axis([0,9,0.3,0.8])
xticks([1,2,4,6,8])
grid on
title('Nominal','Interpreter','latex')
xlabel('$R_{min}$','Interpreter','latex')
ylabel('Silhouette Score', 'Interpreter','latex')

subplot(2,3,2)
errorbar(R_min_vec, acc_central_gro, std_central_gro,trunc_std_central_gro,...
    'Color',[0.3010 0.7450 0.9330],'LineWidth',1)
hold on
errorbar(R_min_vec, acc_clem_gro, std_clem_gro, trunc_std_clem_gro,...
    'Color',[0.4940 0.1840 0.5560],'LineWidth',1)
axis([0,9,0.3,0.8])
xticks([1,2,4,6,8])
grid on
title('Client Imbalance','Interpreter','latex')
xlabel('$R_{min}$','Interpreter','latex')
ylabel('Silhouette Score', 'Interpreter','latex')

subplot(2,3,3)
errorbar(R_min_vec, acc_central_cla, std_central_cla, trunc_std_central_cla,...
    'Color',[0.3010 0.7450 0.9330],'LineWidth',1)
hold on
errorbar(R_min_vec, acc_clem_cla, std_clem_cla, trunc_std_clem_nom,...
    'Color',[0.4940 0.1840 0.5560],'LineWidth',1)
axis([0,9,0.3,0.8])
xticks([1,2,4,6,8])
grid on
title('Cluster Imbalance','Interpreter','latex')
xlabel('$R_{min}$','Interpreter','latex')
ylabel('Silhouette Score', 'Interpreter','latex')
legend({'GMM (Central)','FedGEM (Ours)'},'Interpreter','latex')

subplot(2,3,4)
plot(true_cl_x,true_cl_y,'k--','LineWidth',1)
hold on
errorbar(R_min_vec,ave_n_cl_nom,std_n_cl_nom,...
    'Color',[0.4940 0.1840 0.5560],'LineWidth',1)
axis([0,9,4,16])
xticks([1,2,4,6,8])
grid on
title('Nominal','Interpreter','latex')
xlabel('$R_{min}$','Interpreter','latex')
ylabel('Number of Clusters', 'Interpreter','latex')

subplot(2,3,5)
plot(true_cl_x,true_cl_y,'k--','LineWidth',1)
hold on
errorbar(R_min_vec,ave_n_cl_gro,std_n_cl_gro,...
    'Color',[0.4940 0.1840 0.5560],'LineWidth',1)
axis([0,9,4,16])
xticks([1,2,4,6,8])
grid on
title('Client Imbalance','Interpreter','latex')
xlabel('$R_{min}$','Interpreter','latex')
ylabel('Number of Clusters', 'Interpreter','latex')

subplot(2,3,6)
plot(true_cl_x,true_cl_y,'k--','LineWidth',1)
hold on
errorbar(R_min_vec,ave_n_cl_cla,std_n_cl_cla,...
    'Color',[0.4940 0.1840 0.5560],'LineWidth',1)
axis([0,9,4,16])
xticks([1,2,4,6,8])
grid on
title('Cluster Imbalance','Interpreter','latex')
xlabel('$R_{min}$','Interpreter','latex')
ylabel('Number of Clusters', 'Interpreter','latex')
legend({'True','Estimated'},'Interpreter','latex')

function [trunc_std_vec] = truncate_std(mean_vec,std_vec)
vec = ones(1,length(mean_vec));
temp_std_vec = vec - mean_vec;
trunc_std_vec = ones(1,length(mean_vec));
for i = 1:length(mean_vec)
    trunc_std_vec(i) = min([temp_std_vec(i),std_vec(i)]);
end
end