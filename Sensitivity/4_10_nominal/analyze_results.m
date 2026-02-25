num_exp = 50;
result_clem_vec = zeros(1,num_exp);
result_central_vec = zeros(1,num_exp);

pred_cluster_vec = zeros(1,num_exp);
true_cluster_vec = zeros(1,num_exp);

for i = 1:num_exp
    filename = sprintf('FedClem_Abl_4_10_%d.mat',i);
    load(filename)
    [maxx,idx] = max(clem_sil_vec);
    result_clem_vec(i) = max(clem_sil_vec);
    result_central_vec(i) = max(central_sil_vec);

    pred_cluster_vec(i) = clem_num_cl_vec(idx);
    true_cluster_vec(i) = true_num_cl;
end

acc_clem_4_10_nom = mean(result_clem_vec);
acc_central_4_10_nom = mean(result_central_vec);

std_clem_4_10_nom = std(result_clem_vec);
std_central_4_10_nom = std(result_central_vec);

ave_n_cl_4_10_nom = mean(pred_cluster_vec);
std_n_cl_4_10_nom = std(pred_cluster_vec);