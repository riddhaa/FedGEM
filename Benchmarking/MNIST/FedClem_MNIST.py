import gurobipy as grb
import numpy as np
import timeit
import scipy
import sklearn
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import silhouette_score
import copy
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from sklearn.cluster import KMeans
import timeit
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptive_preprocessing import preprocess_clients


def gen_data(perc_train, G, clusters_per_client=-1):
    
    x_train = np.load('embeddings_vae_train.npy')
    x_test = np.load('embeddings_vae_test.npy')

    y_train = np.load('labels_train.npy')
    y_test = np.load('labels_test.npy')
    
    x_all = np.concatenate([x_train,x_test],axis=0)
    y_all = np.concatenate([y_train,y_test])
    
    _, n_features = x_all.shape
        
    n_train_total = int(perc_train*len(x_all))
    n_test_total = len(x_all) - n_train_total
    
    n_clusters = len(np.unique(y_all))
    
    
    data_dict = {}
    cluster_weights = np.zeros(n_clusters)
    for cl in range(n_clusters):
        data_dict['x_'+str(cl)] = x_all[y_all == cl]
        data_dict['y_'+str(cl)] = y_all[y_all == cl]
        cluster_weights[cl] = np.sum(y_all == cl)/len(y_all)
        
    print(cluster_weights)
    
    if any(clusters_per_client) == -1:
        clusters_per_client = np.ones(G)*n_clusters
        
    clusters_per_client = clusters_per_client.astype(int)
        
    train_clients = {}
    test_clients = {}
    centroids_clients = {}
    counters = {}
    
    for g in range(G):
        n_train = int((1/G)*n_train_total)
        n_test = int((1/G)*n_test_total)
        clusters_g = np.random.choice(n_clusters, size=clusters_per_client[g], replace=False)
        print(clusters_g)
        centroids_g = np.zeros([len(clusters_g),n_features])
        centroids_g[0] = np.mean(x_all[y_all == clusters_g[0]])
        
        weights_g = cluster_weights[clusters_g]
        
        n_train_0 = int(weights_g[0]*n_train)
        n_test_0 = int(weights_g[0]*n_test)
        
        x_curr = data_dict['x_'+str(clusters_g[0])]
        y_curr = data_dict['y_'+str(clusters_g[0])]
        
        x_train_g = x_curr[:n_train_0]
        x_test_g = x_curr[n_train_0:n_train_0+n_test_0]
        y_test_g = y_curr[n_train_0:n_train_0+n_test_0]
        
        x_curr = x_curr[n_train_0 + n_test_0:]
        y_curr = y_curr[n_train_0 + n_test_0:]
        
        data_dict['x_'+str(clusters_g[0])] = x_curr
        data_dict['y_'+str(clusters_g[0])] = y_curr
        
        for c in range(1,len(clusters_g)):
            centroids_g[c] = np.mean(x_all[y_all == clusters_g[c]])
            
            n_train_c = int(weights_g[c]*n_train)
            n_test_c = int(weights_g[c]*n_test)
            
            x_curr = data_dict['x_'+str(clusters_g[c])]
            y_curr = data_dict['y_'+str(clusters_g[c])]
            
            x_train_g = np.concatenate([x_train_g,x_curr[:n_train_c]],axis=0)
            x_test_g = np.concatenate([x_test_g,x_curr[n_train_c:n_train_c+n_test_c]],axis=0)
            y_test_g = np.concatenate([y_test_g,y_curr[n_train_c:n_train_c+n_test_c]],axis=0)
            
            x_curr = x_curr[n_train_c + n_test_c:]
            y_curr = y_curr[n_train_c + n_test_c:]
            
            data_dict['x_'+str(clusters_g[c])] = x_curr
            data_dict['y_'+str(clusters_g[c])] = y_curr
            
            
        train_clients['client_'+str(g)] = x_train_g
        test_clients['x_'+str(g)] = x_test_g
        test_clients['y_'+str(g)] = y_test_g
        centroids_clients['client_'+str(g)] = centroids_g
        
    train_central = train_clients['client_'+str(0)]
    x_test_central = test_clients['x_'+str(0)]
    y_test_central = test_clients['y_'+str(0)]
    
    for g in range(1,G):
        train_central = np.concatenate([train_central,train_clients['client_'+str(g)]],axis=0)
        x_test_central = np.concatenate([x_test_central,test_clients['x_'+str(g)]],axis=0)
        y_test_central = np.concatenate([y_test_central,test_clients['y_'+str(g)]],axis=0)
        
    return train_clients, test_clients, centroids_clients, train_central, x_test_central, y_test_central


def kmeans_pp(X, k, random_state=None):
    np.random.seed(random_state)
    n_samples, _ = X.shape
    centroids = []

    first_idx = np.random.randint(n_samples)
    centroids.append(X[first_idx])

    for _ in range(1, k):
        distances = np.array([min(np.linalg.norm(x - c)**2 for c in centroids) for x in X])
        probabilities = distances / distances.sum()
        cumulative_probs = np.cumsum(probabilities)
        r = np.random.rand()

        next_idx = np.searchsorted(cumulative_probs, r)
        centroids.append(X[next_idx])

    return np.array(centroids)

def initialize_clients(train_clients,cpc):
    Theta_init = {}
    
    for g in range(len(train_clients)):
        centroids_g = kmeans_pp(train_clients['client_'+str(g)],cpc[g])
        Theta_init['Theta_'+str(g)] = centroids_g
        
    return Theta_init

def true_unique_clusters(client_centroids):
    X = client_centroids['client_0']
    
    for g in range(1,len(client_centroids)):
        X = np.concatenate([X,client_centroids['client_'+str(g)]],axis=0)
        
    return len(np.unique(X,axis=0))


class FedClem:
    
    def __init__(self,param,vars_init):
        self.rad_final_factor = param['rad']
        self.local_steps = param['local_steps']
        self.t_steps = param['t_steps']
        self.max_iter = param['max_iter']
        self.aggregate_final = param['aggregate_final']
        self.Theta_curr = copy.deepcopy(vars_init['theta_dict'])
        self.G = len(self.Theta_curr)
        self.Pi = vars_init['pi_dict']
        self.P = param['P']
        
    def comp_M_theta(self,theta_all,pi_all,local_data):
        N = len(local_data)
        K = len(theta_all)
        rvs = {}
        for j in range(K):
            rvs['rv_'+str(j)] = scipy.stats.multivariate_normal(mean=theta_all[j], cov=np.identity(self.P))
            
        denom_vec = np.zeros(N)
        for n in range(N):
            val_n = 0
            for j in range(K):
                val_n += pi_all[j]*rvs['rv_'+str(j)].pdf(local_data[n])
            denom_vec[n] = val_n
        
        M_theta_all = np.zeros(theta_all.shape)
        
        for k in range(K):
            gamma_i_k = np.zeros(N)
            num_i_k = np.zeros([N,self.P])
            for n in range(N):
                gamma_i_k[n] = (pi_all[k]*rvs['rv_'+str(k)].pdf(local_data[n]))/denom_vec[n]
                num_i_k[n] = gamma_i_k[n]*local_data[n]
                
            M_theta_all[k] = (1/np.sum(gamma_i_k))*np.sum(num_i_k,axis=0)
            
        return M_theta_all
        
        
    def rad_feasibility_prob(self,theta_all,pi_all,M_theta_k,eps,local_data,k):
        N = len(local_data)
        theta_k = theta_all[k]
        pi_k = pi_all[k]
        K = len(theta_all)
        rvs = {}
        for j in range(K):
            rvs['rv_'+str(j)] = scipy.stats.multivariate_normal(mean=theta_all[j], cov=np.identity(self.P))
            
        denom_vec = np.zeros(N)
        for n in range(N):
            val_n = 0
            for j in range(K):
                val_n += pi_all[j]*rvs['rv_'+str(j)].pdf(local_data[n])
            denom_vec[n] = val_n

            
        term_2 = 0
        term_4a = 0
        sum_gamma = 0
        for n in range(N):
            sum_gamma += (pi_k*rvs['rv_'+str(k)].pdf(local_data[n]))/denom_vec[n]
            
            gamma_i_k =  (pi_k*rvs['rv_'+str(k)].pdf(local_data[n]))/denom_vec[n]
            mult_temp = np.linalg.norm(local_data[n] - M_theta_k)**2 - np.linalg.norm(local_data[n] - theta_k)**2 - eps
            term_2 += gamma_i_k*mult_temp
            
            mult_temp2 = np.linalg.norm(local_data[n] - theta_k)**2
            term_4a += gamma_i_k*mult_temp2
            
        term_4 = sum_gamma*term_4a
        
        model = grb.Model()
        model.setParam('OutputFlag',False)
        
        var_lambda = model.addVar(vtype=grb.GRB.CONTINUOUS,lb=sum_gamma)
        var_t = model.addVar(vtype=grb.GRB.CONTINUOUS,lb=0)
        
        
        model.update()
        model.addQConstr(eps*var_lambda*var_lambda + term_2*var_lambda - var_t + term_4 <= 0)
        
        obj = var_t
        model.setObjective(obj,grb.GRB.MINIMIZE)
        
        model.optimize()
        
        t_opt = var_t.x
        
        return t_opt
    
    def obtain_radius(self,theta_all,pi_all,M_theta_k,local_data,k):
        theta_k = theta_all[k]
        eps_lb = 0
        eps_ub = np.linalg.norm(M_theta_k - theta_k)**2
        eps_curr = (eps_ub + eps_lb)/2
        eps_prev = eps_curr
        diff = 10
        l = 0
        
        while diff <= 1e-7 and l < self.t_steps:
            t_curr = self.rad_feasibility_prob(theta_all,pi_all,M_theta_k,eps_curr,local_data,k)
            
            if t_curr <= 1e-10:
                eps_lb = copy.deepcopy(eps_curr)
                eps_curr = (eps_ub + eps_lb)/2
                
            else:
                eps_ub = copy.deepcopy(eps_curr)
                eps_curr = (eps_ub + eps_lb)/2
                
            diff = np.abs(eps_curr - eps_prev)
            
            eps_prev = eps_curr
            l += 1
              
        return eps_curr
    
    def train_client_g(self,local_data,g):
        theta_all = copy.deepcopy(self.Theta_curr['Theta_'+str(g)])
        pi_all = self.Pi['Pi_'+str(g)]
        K = len(theta_all)
        eps_vec = np.zeros(K)
        
        for s in range(self.local_steps):
            M_theta_all = self.comp_M_theta(theta_all,pi_all,local_data)

            if self.local_steps - 1 > s:
                theta_all = M_theta_all
                self.Theta_curr['Theta_'+str(g)] = theta_all
                
        for k in range(K):
            M_theta_k = M_theta_all[k]
            eps_k = self.obtain_radius(theta_all,pi_all,M_theta_k,local_data,k)
            eps_vec[k] = eps_k
            
        return M_theta_all, eps_vec
    
    def train_all_clients(self,all_data):
        amb_sets_all = {}
        for g in range(self.G):
            amb_sets_local = {}
            local_data = all_data['client_'+str(g)]
            M_theta_all, eps_vec = self.train_client_g(local_data,g)
            
            for k in range(len(eps_vec)):
                amb_sets_local['centroid_'+str(k)] = M_theta_all[k]
                amb_sets_local['radius_'+str(k)] = eps_vec[k]
                amb_sets_local['assigned_'+str(k)] = False
                amb_sets_local['cluster_'+str(k)] = -1
                amb_sets_local['estimates_'+str(k)] = np.ones([1,len(M_theta_all[k])])
                amb_sets_local['estimates_'+str(k)][0] = M_theta_all[k]
                
            amb_sets_all['client_'+str(g)] = amb_sets_local
            
        return amb_sets_all
    
    def server_comp_clustering(self,amb_sets_all):
        clusters = []
        for g1 in range(self.G):
            curr_client1 = copy.deepcopy(amb_sets_all['client_'+str(g1)])
            K_curr1 = int(len(curr_client1)/5)
            for k1 in range(K_curr1):
                for g2 in range(g1+1,G):
                    curr_client2 = copy.deepcopy(amb_sets_all['client_'+str(g2)])
                    K_curr2 = int(len(curr_client2)/5)
                    for k2 in range(K_curr2):
                        curr_client1 = copy.deepcopy(amb_sets_all['client_'+str(g1)])
                        curr_client2 = copy.deepcopy(amb_sets_all['client_'+str(g2)])
                        
                        theta_1 = curr_client1['centroid_'+str(k1)]
                        eps_1 = curr_client1['radius_'+str(k1)]
                        
                        theta_2 = curr_client2['centroid_'+str(k2)]
                        eps_2 = curr_client2['radius_'+str(k2)]
                        
                        theta_opt_vec = self.intersection_problem(theta_1,eps_1,theta_2,eps_2)
                        
                        if all(theta_opt_vec != np.ones(theta_1.shape)*float('inf')):
                            theta_opt = np.ones([1,self.P])
                            theta_opt[0,:] = theta_opt_vec
                            
                            estimates_1 = copy.deepcopy(curr_client1['estimates_'+str(k1)])
                            estimates_1 = np.concatenate([estimates_1,theta_opt],axis=0)
                            curr_client1['estimates_'+str(k1)] = estimates_1
                            amb_sets_all['client_'+str(g1)] = curr_client1
                            
                            estimates_2 = copy.deepcopy(curr_client2['estimates_'+str(k2)])
                            estimates_2 = np.concatenate([estimates_2,theta_opt],axis=0)
                            curr_client2['estimates_'+str(k2)] = estimates_2
                            amb_sets_all['client_'+str(g2)] = curr_client2
                            
                            assigned_1 = curr_client1['assigned_'+str(k1)]
                            cluster_num_1 = curr_client1['cluster_'+str(k1)]
                            
                            assigned_2 = curr_client2['assigned_'+str(k2)]
                            cluster_num_2 = curr_client2['cluster_'+str(k2)]
                            
                            if assigned_1 == False and assigned_2 == True:
                                can_assign = self.cluster_sanity_check(clusters,cluster_num_2,g1)
                                if can_assign:
                                    clusters[cluster_num_2].append(str(g1)+'-'+str(k1))
                                    curr_client1['assigned_'+str(k1)] = True
                                    curr_client1['cluster_'+str(k1)] = cluster_num_2
                                    amb_sets_all['client_'+str(g1)] = curr_client1
                                    
                                    estimates_1 = copy.deepcopy(curr_client1['estimates_'+str(k1)])
                                    estimates_1 = np.concatenate([estimates_1,theta_opt],axis=0)
                                    curr_client1['estimates_'+str(k1)] = estimates_1
                                    amb_sets_all['client_'+str(g1)] = curr_client1

                                    estimates_2 = copy.deepcopy(curr_client2['estimates_'+str(k2)])
                                    estimates_2 = np.concatenate([estimates_2,theta_opt],axis=0)
                                    curr_client2['estimates_'+str(k2)] = estimates_2
                                    amb_sets_all['client_'+str(g2)] = curr_client2
                                
                            elif assigned_1 == True and assigned_2 == False:
                                can_assign = self.cluster_sanity_check(clusters,cluster_num_1,g2)
                                if can_assign:
                                    clusters[cluster_num_1].append(str(g2)+'-'+str(k2))
                                    curr_client2['assigned_'+str(k2)] = True
                                    curr_client2['cluster_'+str(k2)] = cluster_num_1
                                    amb_sets_all['client_'+str(g2)] = curr_client2
                                    
                                    estimates_1 = copy.deepcopy(curr_client1['estimates_'+str(k1)])
                                    estimates_1 = np.concatenate([estimates_1,theta_opt],axis=0)
                                    curr_client1['estimates_'+str(k1)] = estimates_1
                                    amb_sets_all['client_'+str(g1)] = curr_client1

                                    estimates_2 = copy.deepcopy(curr_client2['estimates_'+str(k2)])
                                    estimates_2 = np.concatenate([estimates_2,theta_opt],axis=0)
                                    curr_client2['estimates_'+str(k2)] = estimates_2
                                    amb_sets_all['client_'+str(g2)] = curr_client2
                                
                            elif assigned_1 == False and assigned_2 == False:
                                new_cluster = [str(g1)+'-'+str(k1),str(g2)+'-'+str(k2)]
                                clusters.append(new_cluster)
                                new_cluster_num = len(clusters) - 1
                                
                                curr_client1['assigned_'+str(k1)] = True
                                curr_client1['cluster_'+str(k1)] = new_cluster_num
                                amb_sets_all['client_'+str(g1)] = curr_client1
                                
                                curr_client2['assigned_'+str(k2)] = True
                                curr_client2['cluster_'+str(k2)] = new_cluster_num
                                amb_sets_all['client_'+str(g2)] = curr_client2
                                
                                estimates_1 = copy.deepcopy(curr_client1['estimates_'+str(k1)])
                                estimates_1 = np.concatenate([estimates_1,theta_opt],axis=0)
                                curr_client1['estimates_'+str(k1)] = estimates_1
                                amb_sets_all['client_'+str(g1)] = curr_client1

                                estimates_2 = copy.deepcopy(curr_client2['estimates_'+str(k2)])
                                estimates_2 = np.concatenate([estimates_2,theta_opt],axis=0)
                                curr_client2['estimates_'+str(k2)] = estimates_2
                                amb_sets_all['client_'+str(g2)] = curr_client2
                                
                            elif assigned_1 == True and assigned_2 == True and cluster_num_1 != cluster_num_2:
                                can_merge = self.cluster_sanity_check2(clusters,cluster_num_1,cluster_num_2)
                                if can_merge:
                                    clusters, amb_sets_all_new = self.reorganize_clusters(
                                        clusters,cluster_num_1,cluster_num_2,amb_sets_all)
                                    amb_sets_all = copy.deepcopy(amb_sets_all_new)
                                    
                                    curr_client1 = copy.deepcopy(amb_sets_all['client_'+str(g1)])
                                    curr_client2 = copy.deepcopy(amb_sets_all['client_'+str(g2)])
                                    
                                    estimates_1 = copy.deepcopy(curr_client1['estimates_'+str(k1)])
                                    estimates_1 = np.concatenate([estimates_1,theta_opt],axis=0)
                                    curr_client1['estimates_'+str(k1)] = estimates_1
                                    amb_sets_all['client_'+str(g1)] = curr_client1

                                    estimates_2 = copy.deepcopy(curr_client2['estimates_'+str(k2)])
                                    estimates_2 = np.concatenate([estimates_2,theta_opt],axis=0)
                                    curr_client2['estimates_'+str(k2)] = estimates_2
                                    amb_sets_all['client_'+str(g2)] = curr_client2
                                
                if curr_client1['assigned_'+str(k1)] == False:
                    clusters.append([str(g1)+'-'+str(k1)])
                    curr_client1['assigned_'+str(k1)] = True
                    curr_client1['cluster_'+str(k1)] = len(clusters) - 1
                    amb_sets_all['client_'+str(g1)] = curr_client1
                    
        return amb_sets_all, clusters
    
    def server_comp_clustering_final(self,amb_sets_all):
        clusters = []
        for g1 in range(self.G):
            curr_client1 = copy.deepcopy(amb_sets_all['client_'+str(g1)])
            K_curr1 = int(len(curr_client1)/5)
            for k1 in range(K_curr1):
                for g2 in range(g1+1,G):
                    curr_client2 = copy.deepcopy(amb_sets_all['client_'+str(g2)])
                    K_curr2 = int(len(curr_client2)/5)
                    for k2 in range(K_curr2):
                        curr_client1 = copy.deepcopy(amb_sets_all['client_'+str(g1)])
                        curr_client2 = copy.deepcopy(amb_sets_all['client_'+str(g2)])
                        
                        theta_1 = curr_client1['centroid_'+str(k1)]
                        eps_1 = curr_client1['radius_'+str(k1)]
                        theta_1_re = np.reshape(theta_1,[1,len(theta_1)])
                        
                        theta_2 = curr_client2['centroid_'+str(k2)]
                        eps_2 = curr_client2['radius_'+str(k2)]
                        theta_2_re = np.reshape(theta_2,[1,len(theta_2)])
                        
                        theta_opt_vec = self.intersection_problem(theta_1,eps_1,theta_2,eps_2)
                        
                        if all(theta_opt_vec != np.ones(theta_1.shape)*float('inf')):
                            theta_opt = np.ones([1,self.P])
                            theta_opt[0,:] = theta_opt_vec
                            
                            assigned_1 = curr_client1['assigned_'+str(k1)]
                            cluster_num_1 = curr_client1['cluster_'+str(k1)]
                            
                            assigned_2 = curr_client2['assigned_'+str(k2)]
                            cluster_num_2 = curr_client2['cluster_'+str(k2)]
                            
                            if assigned_1 == False and assigned_2 == True:
                                can_assign = self.cluster_sanity_check(clusters,cluster_num_2,g1)
                                if can_assign:
                                    clusters[cluster_num_2].append(str(g1)+'-'+str(k1))
                                    curr_client1['assigned_'+str(k1)] = True
                                    curr_client1['cluster_'+str(k1)] = cluster_num_2
                                    amb_sets_all['client_'+str(g1)] = curr_client1
                                    
                                    estimates_1 = copy.deepcopy(curr_client1['estimates_'+str(k1)])
                                    estimates_2 = copy.deepcopy(curr_client2['estimates_'+str(k2)])
                                    
                                    estimates_1 = np.concatenate([estimates_1,estimates_2],axis=0)
                                    
                                    curr_client1['estimates_'+str(k1)] = estimates_1
                                    amb_sets_all['client_'+str(g1)] = curr_client1

                                    
                                    for c in range(len(clusters[cluster_num_2])):
                                        g_temp, k_temp = map(int, clusters[cluster_num_2][c].split('-'))
                                        if g_temp != g1:
                                            amb_sets_all['client_'+str(g_temp)]['estimates_'+str(k_temp)] = np.concatenate(
                                            [amb_sets_all['client_'+str(g_temp)]['estimates_'+str(k_temp)],
                                            theta_1_re],axis=0)

                                
                            elif assigned_1 == True and assigned_2 == False:
                                can_assign = self.cluster_sanity_check(clusters,cluster_num_1,g2)
                                if can_assign:
                                    clusters[cluster_num_1].append(str(g2)+'-'+str(k2))
                                    curr_client2['assigned_'+str(k2)] = True
                                    curr_client2['cluster_'+str(k2)] = cluster_num_1
                                    amb_sets_all['client_'+str(g2)] = curr_client2
                                    
                                    estimates_1 = copy.deepcopy(curr_client1['estimates_'+str(k1)])
                                    estimates_2 = copy.deepcopy(curr_client2['estimates_'+str(k2)])
                                    
                                    estimates_2 = np.concatenate([estimates_2,estimates_1],axis=0)
                                    
                                    
                                    curr_client2['estimates_'+str(k2)] = estimates_2
                                    amb_sets_all['client_'+str(g2)] = curr_client2
                                    
                                    for c in range(len(clusters[cluster_num_1])):
                                        g_temp, k_temp = map(int, clusters[cluster_num_1][c].split('-'))
                                        if g_temp != g2:
                                            amb_sets_all['client_'+str(g_temp)]['estimates_'+str(k_temp)] = np.concatenate(
                                            [amb_sets_all['client_'+str(g_temp)]['estimates_'+str(k_temp)],
                                            theta_2_re],axis=0)
                                
                            elif assigned_1 == False and assigned_2 == False:
                                new_cluster = [str(g1)+'-'+str(k1),str(g2)+'-'+str(k2)]
                                clusters.append(new_cluster)
                                new_cluster_num = len(clusters) - 1
                                
                                curr_client1['assigned_'+str(k1)] = True
                                curr_client1['cluster_'+str(k1)] = new_cluster_num
                                amb_sets_all['client_'+str(g1)] = curr_client1
                                
                                curr_client2['assigned_'+str(k2)] = True
                                curr_client2['cluster_'+str(k2)] = new_cluster_num
                                amb_sets_all['client_'+str(g2)] = curr_client2
                                
                                estimates_1 = copy.deepcopy(curr_client1['estimates_'+str(k1)])
                                estimates_1 = np.concatenate([estimates_1,theta_2_re],axis=0)
                                curr_client1['estimates_'+str(k1)] = estimates_1
                                amb_sets_all['client_'+str(g1)] = curr_client1

                                estimates_2 = copy.deepcopy(curr_client2['estimates_'+str(k2)])
                                estimates_2 = np.concatenate([estimates_2,theta_1_re],axis=0)
                                curr_client2['estimates_'+str(k2)] = estimates_2
                                amb_sets_all['client_'+str(g2)] = curr_client2
                                
                            elif assigned_1 == True and assigned_2 == True and cluster_num_1 != cluster_num_2:
                                can_merge = self.cluster_sanity_check2(clusters,cluster_num_1,cluster_num_2)
                                if can_merge:
                                    estimates_1 = copy.deepcopy(curr_client1['estimates_'+str(k1)])
                                    estimates_2 = copy.deepcopy(curr_client2['estimates_'+str(k2)])
                                    
                                    for c in range(len(clusters[cluster_num_1])):
                                        g_temp, k_temp = map(int, clusters[cluster_num_1][c].split('-'))
                                        amb_sets_all['client_'+str(g_temp)]['estimates_'+str(k_temp)] = np.concatenate(
                                        [amb_sets_all['client_'+str(g_temp)]['estimates_'+str(k_temp)],
                                        estimates_2],axis=0)
                                     
                                    for c in range(len(clusters[cluster_num_2])):
                                        g_temp, k_temp = map(int, clusters[cluster_num_2][c].split('-'))
                                        amb_sets_all['client_'+str(g_temp)]['estimates_'+str(k_temp)] = np.concatenate(
                                        [amb_sets_all['client_'+str(g_temp)]['estimates_'+str(k_temp)],
                                        estimates_1],axis=0)
                                    
                                    clusters, amb_sets_all_new = self.reorganize_clusters(
                                        clusters,cluster_num_1,cluster_num_2,amb_sets_all)
                                    amb_sets_all = copy.deepcopy(amb_sets_all_new)
                                    
                                    curr_client1 = copy.deepcopy(amb_sets_all['client_'+str(g1)])
                                    curr_client2 = copy.deepcopy(amb_sets_all['client_'+str(g2)])

                                
                if curr_client1['assigned_'+str(k1)] == False:
                    clusters.append([str(g1)+'-'+str(k1)])
                    curr_client1['assigned_'+str(k1)] = True
                    curr_client1['cluster_'+str(k1)] = len(clusters) - 1
                    amb_sets_all['client_'+str(g1)] = curr_client1
                    
        return amb_sets_all, clusters
    
    def server_comp_aggregation(self,amb_sets_all):
        for g in range(self.G):
            theta_g = self.Theta_curr['Theta_'+str(g)]
            K = len(theta_g)
            for k in range(K):
                curr_client = amb_sets_all['client_'+str(g)]
                new_estimate = np.mean(curr_client['estimates_'+str(k)],axis=0)
                theta_g[k] = new_estimate
                
            self.Theta_curr['Theta_'+str(g)] = theta_g
                        
                                
    def reorganize_clusters(self,clusters,cluster_num_1,cluster_num_2,amb_sets_all):
        cluster_1 = clusters[cluster_num_1]
        cluster_2 = clusters[cluster_num_2]
        
        for c in range(len(cluster_2)):
            component = cluster_2[c]
            cluster_1.append(component)
            g, k = map(int, component.split('-'))
            curr_client = copy.deepcopy(amb_sets_all['client_'+str(g)])
            curr_client['cluster_'+str(k)] = cluster_num_1
            amb_sets_all['client_'+str(g)] = curr_client
            
        clusters[cluster_num_1] = cluster_1
        clusters.pop(cluster_num_2)
        
        if cluster_num_2 < len(clusters):
            for c in range(cluster_num_2,len(clusters)):
                curr_cluster = clusters[c]
                for d in range(len(curr_cluster)):
                    component = curr_cluster[d]
                    g, k = map(int, component.split('-'))
                    curr_client = copy.deepcopy(amb_sets_all['client_'+str(g)])
                    curr_client['cluster_'+str(k)] = c
                    amb_sets_all['client_'+str(g)] = curr_client
                    
        return clusters, amb_sets_all
                    
        
    def intersection_problem(self,theta_1,eps_1,theta_2,eps_2):
        
        if np.linalg.norm(theta_1 - theta_2) > np.sqrt(eps_1) + np.sqrt(eps_2):
            theta_opt = np.ones(theta_1.shape)*float('inf')
        else:
            w = np.linalg.norm(theta_1 - theta_2)
            val = np.clip(0.5, 1 - (np.sqrt(eps_2)/w), (np.sqrt(eps_1)/w))
            theta_opt = theta_1 + val*(theta_1 - theta_2)
        
        return theta_opt
    
    def final_aggregation(self,clusters):
        self.clustered_centroids = np.zeros([len(clusters),self.P])
        for c in range(len(clusters)):
            curr_cluster = clusters[c]
            centroids_arr = np.zeros([len(curr_cluster),self.P])
            for d in range(len(curr_cluster)):
                component = curr_cluster[d]
                g, k = map(int, component.split('-'))
                centroids_arr[d] = self.Theta_curr['Theta_'+str(g)][k]
                
            self.clustered_centroids[c] = np.mean(centroids_arr,axis=0)
            
    def check_converged(self,Theta_prev):
        cont_tr = False
        
        for g in range(len(Theta_prev)):
            theta_g = self.Theta_curr['Theta_'+str(g)]
            theta_prev_g = Theta_prev['Theta_'+str(g)]
            
            for k in range(len(theta_g)):
                if np.linalg.norm(theta_g[k] - theta_prev_g[k]) > 1e-3:
                    cont_tr = True
                    break
        return cont_tr
        
        
    def train(self,train_data):
        l = 0
        amb_sets_all = self.train_all_clients(train_data)
        amb_sets_all, clusters = self.server_comp_clustering(amb_sets_all)
        print(len(clusters))
        self.server_comp_aggregation(amb_sets_all)
        print(l)
        Theta_prev = copy.deepcopy(self.Theta_curr)
        continue_training = True
        while continue_training and l < self.max_iter:
            amb_sets_all = self.train_all_clients(train_data)
            amb_sets_all, clusters = self.server_comp_clustering(amb_sets_all)
            print(len(clusters))
            self.server_comp_aggregation(amb_sets_all)
            continue_training = self.check_converged(Theta_prev)
            Theta_prev = copy.deepcopy(self.Theta_curr)
            l += 1
            print(l)
            
        amb_sets_all = self.train_all_clients(train_data)
        R_min_vec = self.comp_R_min_curr()
        for g in range(self.G):
            K_curr = int(len(amb_sets_all['client_'+str(g)])/5)
            for k in range(K_curr):
                R_g = R_min_vec[g]
                pi_g_k = self.Pi['Pi_'+str(g)][k]
                N_g = len(train_data['client_'+str(g)])
                rad_g_k = self.rad_final_factor*(R_g/(pi_g_k*np.sqrt(N_g)))
                amb_sets_all['client_'+str(g)]['radius_'+str(k)] = rad_g_k
                
        amb_sets_all, clusters = self.server_comp_clustering_final(amb_sets_all)
        self.server_comp_aggregation(amb_sets_all)
            
        return self.Theta_curr, clusters
    
    def comp_R_min_curr(self):
        R_min_vec = np.ones(self.G)*float('inf')
        for g in range(self.G):
            theta_g = self.Theta_curr['Theta_'+str(g)]
            K_g = len(theta_g)
            for k1 in range(K_g-1):
                for k2 in range(k1+1,K_g):
                    R_curr = np.linalg.norm(theta_g[k1] - theta_g[k2])
                    if R_curr < R_min_vec[g]:
                        R_min_vec[g] = R_curr
                        
        return R_min_vec
    
    def cluster_sanity_check(self,clusters,cluster_num,g):
        cluster_curr = clusters[cluster_num]
        
        can_assign = True
        
        for c in range(len(cluster_curr)):
            g_curr, k_curr = map(int, cluster_curr[c].split('-'))
            if g_curr == g:
                can_assign = False
                break
                    
        return can_assign
    
    def cluster_sanity_check2(self,clusters,cluster_num_1,cluster_num_2):

        cluster_1 = clusters[cluster_num_1]
        cluster_2 = clusters[cluster_num_2]
        
        can_merge = True
        
        for c1 in range(len(cluster_1)):
            for c2 in range(len(cluster_2)):
                g1, k1 = map(int, cluster_1[c1].split('-'))
                g2, k2 = map(int, cluster_2[c2].split('-'))
                if g1 == g2:
                    can_merge = False
                    break
                    
        return can_merge
        
    
    def test(self,test_data):
        final_pred = self.test_distributed(test_data)
        ar_score = np.zeros(self.G)
        sil_score = np.zeros(self.G)
        for g in range(self.G):
            ar_score[g] = adjusted_rand_score(test_data['y_'+str(g)],final_pred['client_'+str(g)])
            sil_score[g] = silhouette_score(test_data['x_'+str(g)],final_pred['client_'+str(g)])

            
            
        return ar_score, sil_score
    
    def validate(self,test_data):
        final_pred = self.test_distributed(test_data)
        ar_score = np.zeros(self.G)
        sil_score = np.zeros(self.G)
        for g in range(self.G):
            sil_score[g] = silhouette_score(test_data['x_'+str(g)],final_pred['client_'+str(g)])

            
        return sil_score
    
    def test_distributed(self,test_data):
        final_pred = {}
        
        for g in range(self.G):
            theta_g = self.Theta_curr['Theta_'+str(g)]
            pi_g = self.Pi['Pi_'+str(g)]
            x_g = test_data['x_'+str(g)]
            final_pred_g = np.zeros(len(x_g))
            
            rvs = {}
            for j in range(len(pi_g)):
                rvs['rv_'+str(j)] = scipy.stats.multivariate_normal(mean=theta_g[j], cov=np.identity(self.P))
                
            for n in range(len(x_g)):
                denom_n = 0
                
                for j in range(len(pi_g)):
                    denom_n += pi_g[j]*rvs['rv_'+str(j)].pdf(x_g[n])
                    
                gamma_vec = np.zeros(len(pi_g))
                for k in range(len(pi_g)):
                    gamma_k = (pi_g[k]*rvs['rv_'+str(k)].pdf(x_g[n]))/denom_n
                    gamma_vec[k] = gamma_k
                    
                pred_cluster_n = np.argmax(gamma_vec)
                final_pred_g[n] = pred_cluster_n
                
            final_pred['client_'+str(g)] = final_pred_g
            
        return final_pred
        
class AFCL:
    def __init__(self,params,vars_init):
        self.k = params['k']
        self.eta = 1e-3
        self.max_iter = params['max_iter']
        self.init_seeds = vars_init['init_seeds']
        self.G = len(vars_init['init_seeds'])
        self.P = len(vars_init['init_seeds']['Theta_0'][0])
        self.theta_vec = np.zeros([self.G])
        self.w = np.ones(self.G)*(1/self.G)
        
    def server_init_global_seeds(self):
        all_client_seeds = self.init_seeds['Theta_0']
        
        for g in range(1,self.G):
            all_client_seeds = np.concatenate([all_client_seeds,self.init_seeds['Theta_'+str(g)]],axis=0)
            
        init_seeds_global = kmeans_pp(all_client_seeds, self.k)
        
        self.M = init_seeds_global
        
    
    def init_s_vecs(self):
        self.s_vecs = {}
        
        for g in range(self.G):
            s_vec_g = np.ones(self.k)
            
            self.s_vecs['client_'+str(g)] = s_vec_g
            
    def update_s_vec(self,Q):
        s_vec = np.zeros(self.k)
        
        for r in range(self.k):
            s_vec[r] = np.sum(Q[:,r])
            
        return s_vec
            
    def comp_Q(self,train_data_g,s_vec):
        Q = np.zeros([len(train_data_g),self.k])
        
        gamma_vec = np.zeros(self.k)
        for r in range(self.k):
            gamma_vec[r] = s_vec[r]/np.sum(s_vec)
            
        for i in range(len(train_data_g)):
            dist_vec = np.zeros(self.k)
            
            for r in range(self.k):
                dist_vec[r] = gamma_vec[r]*np.linalg.norm(train_data_g[i] - self.M[r])**2
                
            minimizer = np.argmin(dist_vec)
            Q[i,minimizer] = 1
            
        return Q
    
    def comp_R(self,train_data_g,Q):
        R = {}
        for r in range(self.k):
            sample = train_data_g[0]
            R['R_'+str(r)] = np.zeros([len(sample)])
            
        for i in range(len(train_data_g)):
            idx = np.argmax(Q[i])
            temp_vec = self.eta*(train_data_g[i] - self.M[idx])
            R['R_'+str(idx)] = np.concatenate([R['R_'+str(idx)],temp_vec],axis=0)
            
        return R
    
    def comp_B(self,Q_g,train_data_g,s_vec):
        B = np.zeros([self.k,self.P])
        
        for r in range(self.k):
            o_r = s_vec[r]
            temp_vec = np.zeros(self.P)
            
            for i in range(len(train_data_g)):
                temp_vec += Q_g[i,r]*train_data_g[i]
                
            B[r] = (1/o_r)*temp_vec
            
        return B
    
    def comp_z(self,B_g,Q_g,train_data_g):
        z = np.zeros(self.k)
        
        for r in range(self.k):
            temp = 0
            for i in range(len(train_data_g)):
                temp += Q_g[i,r]*np.linalg.norm(B_g[r] - train_data_g[i])**2
                
            z[r] = temp
            
        return z
            
    
    def client_step(self,g,train_data_g):
        self.theta_vec[g] += 1
        s_vec_curr = self.s_vecs['client_'+str(g)]
        
        Q_g = self.comp_Q(train_data_g,s_vec_curr)
        s_vec_new = self.update_s_vec(Q_g)
        
        self.s_vecs['client_'+str(g)] = s_vec_new
        
        R_g = self.comp_R(train_data_g,Q_g)
        
        B_g = self.comp_B(Q_g,train_data_g,s_vec_new)
        
        z_g = self.comp_z(B_g,Q_g,train_data_g)
        
        
        return Q_g, R_g, B_g, z_g
    
    def comp_b_cent(self,B_all):
        B_cent = np.zeros([self.k,self.P])
        
        for r in range(self.k):
            o_r = 0
            
            for g in range(self.G):
                o_r += self.s_vecs['client_'+str(g)][r]
                
            b_r = np.zeros(self.P)
            
            for g in range(self.G):
                b_r += (1/o_r)*self.w[g]*self.s_vecs['client_'+str(g)][r]*B_all['client_'+str(g)][r]
                
            B_cent[r] = b_r
            
        return B_cent
    
    def comp_z_cent(self,z_all):
        z_cent = np.zeros(self.k)
        
        for r in range(self.k):
            o_r = 0
            
            for g in range(self.G):
                o_r += self.s_vecs['client_'+str(g)][r]
                
            z_r = 0
            
            for g in range(self.G):
                z_r += (1/o_r)*self.w[g]*self.s_vecs['client_'+str(g)][r]*z_all['client_'+str(g)][r]
                
            z_cent[r] = z_r
            
        return z_cent
            
    def comp_Z(self,z_cent,B_cent):
        Z = 0
        for l in range(self.k):
            temp_vec = np.zeros(self.k)
            
            for r in range(self.k):
                if r != l:
                    temp_vec[r] = (z_cent[l] + z_cent[r])/(np.linalg.norm(B_cent[l] - B_cent[r])**2)
                    
            Z += np.max(temp_vec)
            
        Z = (1/self.k)*Z
        
        return Z
    
    def update_M(self,R_all):
        new_M = copy.deepcopy(self.M)
        for g in range(self.G):
            M_g = copy.deepcopy(new_M)
            M_g_new = copy.deepcopy(new_M)
            
            for r in range(self.k):
                R_g_r = R_all['client_'+str(g)]['R_'+str(r)]
                for i in range(1,len(R_g_r)):
                    val = (1/(self.eta**2))*np.linalg.norm(self.w[g]*R_g_r[i])**2
                        
                    for l in range(self.k):
                        dist = np.linalg.norm(M_g[r] - M_g[l])**2
                        if dist <= val and l != r:
                                M_g_new[l] = M_g[l] + self.w[g]*R_g_r[i] + self.w[g]*self.eta*(M_g[r] - M_g[l])
                            
            new_M = copy.deepcopy(M_g_new)
            
        self.M = copy.deepcopy(new_M)

    def server_comp(self,Q_all,R_all,B_all,z_all):
        B_cent = self.comp_b_cent(B_all)
        z_cent = self.comp_z_cent(z_all)
        
        Z = self.comp_Z(z_cent,B_cent)
        
        self.update_M(R_all)
    
    
    def train(self,train_data):
        self.server_init_global_seeds()
        self.init_s_vecs()
        
        for _ in range(self.max_iter):
            Q_all = {}
            R_all = {}
            B_all = {}
            z_all = {}
            for g in range(self.G):
                Q_g, R_g, B_g, z_g = self.client_step(g,train_data['client_'+str(g)])
                Q_all['client_'+str(g)] = Q_g
                R_all['client_'+str(g)] = R_g
                B_all['client_'+str(g)] = B_g
                z_all['client_'+str(g)] = z_g
                
            self.server_comp
            
        Q_global = Q_all['client_'+str(0)]
        
        for g in range(1,self.G):
            Q_global = np.concatenate([Q_global,Q_all['client_'+str(g)]],axis=0)
            
        
            
        return self.M,Q_global
    
    def test(self,test_data):
        pred_vec = np.zeros(len(test_data['x']))
        
        for i in range(len(test_data['x'])):
            dist_vec = np.zeros(self.k)
            for r in range(self.k):
                dist_vec[r] = np.linalg.norm(test_data['x'][i] - self.M[r])
                
            pred_vec[i] = np.argmin(dist_vec)
            
        
                
        ar_score = adjusted_rand_score(test_data['y'],pred_vec)
        sil_score = silhouette_score(test_data['x'],pred_vec)
        
        return ar_score, sil_score
        
        
        
    
class k_FED:
    def __init__(self,params,vars_init):
        self.K = params['K']
        self.Theta_init = vars_init['theta']
        self.G = len(self.Theta_init)
        self.P = len(self.Theta_init['Theta_0'][0])
        
    def train_clients(self,train_data):
        self.Theta_clients = {}
        
        for g in range(self.G):
            init_g = self.Theta_init['Theta_'+str(g)]
            n_clusters_g = len(init_g)
            train_data_g = train_data['client_'+str(g)]
            k_means_g = KMeans(n_clusters=n_clusters_g,
                              init=init_g,
                              n_init=1)
            k_means_g.fit(train_data_g)
            centroids_g = k_means_g.cluster_centers_
            self.Theta_clients['Theta_'+str(g)] = centroids_g
            
    def server_comp(self):
        g_curr = np.random.randint(low=0,
                                  high=G)
        
        M = self.Theta_clients['Theta_'+str(g_curr)]
        
        while len(M) < self.K:
            max_dist = 0
            for g in range(self.G):
                if g != g_curr:
                    centroids_g = self.Theta_clients['Theta_'+str(g)]
                    for k_g in range(len(centroids_g)):
                        curr_centroid = centroids_g[k_g]
                        dist_vec = np.zeros(len(M))
                        for k_M in range(len(M)):
                            M_centroid = M[k_M]
                            dist_vec[k_M] = np.linalg.norm(curr_centroid - M_centroid)
                            
                        mean_dist = np.mean(dist_vec)
                        if mean_dist > max_dist:
                            max_dist = mean_dist
                            centroid_temp = np.reshape(curr_centroid, [1,len(curr_centroid)])
            M = np.concatenate([M,centroid_temp],axis=0)
                            
        all_centroids = copy.deepcopy(self.Theta_clients['Theta_'+str(0)])
        for g in range(1,self.G):
            all_centroids = np.concatenate([all_centroids,self.Theta_clients['Theta_'+str(g)]],axis=0)
                            
        k_means_central = KMeans(n_clusters=self.K,
                                init=M,
                                n_init=1,
                                max_iter=1)
        k_means_central.fit(all_centroids)
        self.centroids_final = k_means_central.cluster_centers_
        
    
    def train(self,train_data):
        self.train_clients(train_data)
        self.server_comp()
        
        return self.centroids_final
    
    def test(self,test_data):
        pred_vec = np.zeros(len(test_data['x']))
        
        for i in range(len(test_data['x'])):
            dist_vec = np.zeros(self.K)
            for k in range(self.K):
                dist_vec[k] = np.linalg.norm(test_data['x'][i] - self.centroids_final[k])
                
            pred_vec[i] = np.argmin(dist_vec)
            
        ar_score = adjusted_rand_score(test_data['y'],pred_vec)
        sil_score = silhouette_score(test_data['x'],pred_vec)
        
        return ar_score, sil_score
    
    
class FedKmeans:
    def __init__(self,params):
        self.K = params['K']
        self.G = params['G']
        self.max_iter = params['max_iter']
        
    def kmeans_pp(X, k, random_state=None):
        np.random.seed(random_state)
        n_samples, _ = X.shape
        centroids = []

        first_idx = np.random.randint(n_samples)
        centroids.append(X[first_idx])

        for _ in range(1, k):
            distances = np.array([min(np.linalg.norm(x - c)**2 for c in centroids) for x in X])
            probabilities = distances / distances.sum()
            cumulative_probs = np.cumsum(probabilities)
            r = np.random.rand()

            next_idx = np.searchsorted(cumulative_probs, r)
            centroids.append(X[next_idx])

        return np.array(centroids)
    
    def initialize_clients(self,train_data):
        self.client_centroids = {}
        self.client_sample_num = {}
        for g in range(self.G):
            train_data_g = train_data['client_'+str(g)]
            centroids_g = kmeans_pp(train_data_g,self.K)
            self.client_centroids['client_'+str(g)] = centroids_g
            
            sample_num_vec_g = np.zeros([self.K,1])
            for i in range(len(train_data_g)):
                pred_vec = np.zeros(self.K)
                
                for k in range(self.K):
                    pred_vec[k] = np.linalg.norm(train_data_g[i] - centroids_g[k])
                    
                sample_num_vec_g[np.argmin(pred_vec)] += 1
                
            self.client_sample_num['client_'+str(g)] = sample_num_vec_g
            
    def server_comp(self):
        all_centroids = self.client_centroids['client_0']
        all_sample_nums = self.client_sample_num['client_0']
        for g in range(1,self.G):
            all_centroids = np.concatenate([all_centroids,self.client_centroids['client_'+str(g)]],axis=0)
            all_sample_nums = np.concatenate([all_sample_nums,self.client_sample_num['client_'+str(g)]],axis=0)
            
        k_means_central = KMeans(n_clusters=self.K,
                                init='k-means++',
                                n_init=1)
        k_means_central.fit(all_centroids,sample_weight=all_sample_nums[:,0])
        self.centroids_global = k_means_central.cluster_centers_
        
    def client_comp(self,train_data):
        for g in range(self.G):
            train_data_g = train_data['client_'+str(g)]
            
            sample_num_vec_g = np.zeros([self.K,1])
            for i in range(len(train_data_g)):
                pred_vec = np.zeros(self.K)
                
                for k in range(self.K):
                    pred_vec[k] = np.linalg.norm(train_data_g[i] - self.centroids_global[k])
                    
                sample_num_vec_g[np.argmin(pred_vec)] += 1
                
            centroids_g = self.centroids_global[sample_num_vec_g[:,0] != 0]
            K_g = len(centroids_g)
            
            k_means_g = KMeans(n_clusters=K_g,
                              init=centroids_g,
                              n_init=1)
            k_means_g.fit(train_data_g)
            new_centroids_g = k_means_g.cluster_centers_
            self.client_centroids['client_'+str(g)] = new_centroids_g
            
            sample_num_vec_g = np.zeros([K_g,1])
            for i in range(len(train_data_g)):
                pred_vec = np.zeros(K_g)
                
                for k in range(K_g):
                    pred_vec[k] = np.linalg.norm(train_data_g[i] - new_centroids_g[k])
                    
                sample_num_vec_g[np.argmin(pred_vec)] += 1
                
            self.client_sample_num['client_'+str(g)] = sample_num_vec_g
            
    def train(self,train_data):
        self.initialize_clients(train_data)
        for _ in range(self.max_iter):
            self.server_comp()
            self.client_comp(train_data)
                        
        return self.centroids_global
    
    def test(self,test_data):
        pred_vec = np.zeros(len(test_data['x']))
        
        for i in range(len(test_data['x'])):
            dist_vec = np.zeros(self.K)
            for k in range(self.K):
                dist_vec[k] = np.linalg.norm(test_data['x'][i] - self.centroids_global[k])
                
            pred_vec[i] = np.argmin(dist_vec)
            
        ar_score = adjusted_rand_score(test_data['y'],pred_vec)
        sil_score = silhouette_score(test_data['x'],pred_vec)
        
        return ar_score, sil_score
    
class FFCM_avg1:
    def __init__(self,params):
        self.K = params['K']
        self.G = params['G']
        self.P = params['P']
        self.max_iter = params['max_iter']
        self.m = 2
        
    def init_centroids(self):
        self.centroids_global = np.random.rand(self.K,self.P)
        
    def client_step(self,train_data_g):
        U_g = np.zeros([len(train_data_g),self.K])
        
        for i in range(len(train_data_g)):
            for j in range(self.K):
                val_j = 0
                for k in range(self.K):
                    numer = np.linalg.norm(train_data_g[i] - self.centroids_global[j])**2
                    denom = np.linalg.norm(train_data_g[i] - self.centroids_global[k])**2
                    temp_val = (numer/denom)**(2/(self.m - 1))
                    val_j += temp_val
                    
                U_g[i,j] = 1/val_j
                
        centroids_g = np.zeros([self.K,self.P])
        supp_weight_g = np.zeros([self.K])
        
        for k in range(self.K):
            denom = np.sum(U_g[:,k])
            vec_temp = np.zeros(self.P)
            for i in range(len(train_data_g)):
                vec_temp += U_g[i,k]*train_data_g[i]
                supp_weight_g[k] += U_g[i,k]**self.m
                
            centroids_g[k] = (1/denom)*vec_temp
            
            
        return centroids_g, supp_weight_g
    
    def train_clients(self,train_data):
        centroids_clients = {}
        supp_weight_clients = {}
        
        for g in range(self.G):
            train_data_g = train_data['client_'+str(g)]
            centroids_clients['client_'+str(g)], supp_weight_clients['client_'+str(g)] = self.client_step(train_data_g)
            
        return centroids_clients, supp_weight_clients
    
    def server_comp(self,centroids_clients,supp_weight_clients):
        new_centroids = np.zeros(self.centroids_global.shape)
        
        for k in range(self.K):
            denom = 0
            numer = np.zeros(self.P)
            
            for g in range(self.G):
                denom += supp_weight_clients['client_'+str(g)][k]
                numer += supp_weight_clients['client_'+str(g)][k]*centroids_clients['client_'+str(g)][k]
                
            new_centroids[k] = (1/denom)*numer
            
        self.centroids_global = copy.deepcopy(new_centroids)
        
    def train(self,train_data):
        self.init_centroids()
        
        for _ in range(self.max_iter):
            centroids_clients, supp_weight_clients = self.train_clients(train_data)
            self.server_comp(centroids_clients, supp_weight_clients)
            
        return self.centroids_global
    
    def test(self,test_data):
        pred_vec = np.zeros(len(test_data['x']))
        
        for i in range(len(test_data['x'])):
            dist_vec = np.zeros(self.K)
            for k in range(self.K):
                dist_vec[k] = np.linalg.norm(test_data['x'][i] - self.centroids_global[k])
                
            pred_vec[i] = np.argmin(dist_vec)
            
        ar_score = adjusted_rand_score(test_data['y'],pred_vec)
        sil_score = silhouette_score(test_data['x'],pred_vec)
        
        return ar_score, sil_score
    
class FFCM_avg2:
    def __init__(self,params):
        self.K = params['K']
        self.G = params['G']
        self.P = params['P']
        self.max_iter = params['max_iter']
        self.m = 2
        
    def init_centroids(self):
        self.centroids_global = np.random.rand(self.K,self.P)
        
    def client_step(self,train_data_g):
        U_g = np.zeros([len(train_data_g),self.K])
        
        for i in range(len(train_data_g)):
            for j in range(self.K):
                val_j = 0
                for k in range(self.K):
                    numer = np.linalg.norm(train_data_g[i] - self.centroids_global[j])**2
                    denom = np.linalg.norm(train_data_g[i] - self.centroids_global[k])**2
                    temp_val = (numer/denom)**(2/(self.m - 1))
                    val_j += temp_val
                    
                U_g[i,j] = 1/val_j
                
        centroids_g = np.zeros([self.K,self.P])
        supp_weight_g = np.zeros([self.K])
        
        for k in range(self.K):
            denom = np.sum(U_g[:,k])
            vec_temp = np.zeros(self.P)
            for i in range(len(train_data_g)):
                vec_temp += U_g[i,k]*train_data_g[i]
                supp_weight_g[k] += U_g[i,k]**self.m
                
            centroids_g[k] = (1/denom)*vec_temp
            
            
        return centroids_g, supp_weight_g
    
    def train_clients(self,train_data):
        centroids_clients = {}
        supp_weight_clients = {}
        
        for g in range(self.G):
            train_data_g = train_data['client_'+str(g)]
            centroids_clients['client_'+str(g)], supp_weight_clients['client_'+str(g)] = self.client_step(train_data_g)
            
        return centroids_clients, supp_weight_clients
    
    def server_comp(self,centroids_clients,supp_weight_clients):
        all_centroids = centroids_clients['client_0']
        
        for g in range(1,self.G):
            all_centroids = np.concatenate([all_centroids,centroids_clients['client_'+str(g)]],axis=0)
            
        k_means_central = KMeans(n_clusters=self.K,
                              init='k-means++',
                              n_init=1)
        k_means_central.fit(all_centroids)
        new_centroids = k_means_central.cluster_centers_
        
        self.centroids_global = copy.deepcopy(new_centroids)
        
    def train(self,train_data):
        self.init_centroids()
        
        for _ in range(self.max_iter):
            centroids_clients, supp_weight_clients = self.train_clients(train_data)
            self.server_comp(centroids_clients, supp_weight_clients)
            
        return self.centroids_global
    
    def test(self,test_data):
        pred_vec = np.zeros(len(test_data['x']))
        
        for i in range(len(test_data['x'])):
            dist_vec = np.zeros(self.K)
            for k in range(self.K):
                dist_vec[k] = np.linalg.norm(test_data['x'][i] - self.centroids_global[k])
                
            pred_vec[i] = np.argmin(dist_vec)
            
        ar_score = adjusted_rand_score(test_data['y'],pred_vec)
        sil_score = silhouette_score(test_data['x'],pred_vec)
        
        return ar_score, sil_score
    
    
def cv_split(client_sets,G,n_folds=5):
    client_train_split = {}
    
    client_val_split = {}
        
    for g in range(G):
        idx = np.random.permutation(len(client_sets['client_'+str(g)]))
        x_g = client_sets['client_'+str(g)][idx]

        kf = sklearn.model_selection.KFold(n_splits=n_folds, shuffle=False)
        
        ct = 0
        for train_index, val_index in kf.split(x_g):
            client_train_split['client_'+str(g)+'_'+str(ct)], client_val_split['x_'+str(g)+'_'+str(ct)] = x_g[train_index], x_g[val_index]
                        
            ct += 1

        
    return client_train_split, client_val_split

n_folds = 3
rad_factor_vec = [1e4,1e5,1e6]
perc_train = 0.7
G = 100
client_weights = np.ones(G)*(1/G)
n_clusters = 10
n_features = 10
cpc = np.random.randint(2,n_clusters-1,G)

Pi_init = {}
for g in range(G):
    n_clusters_g = int(cpc[g])
    Pi_init['Pi_'+str(g)] = np.ones(n_clusters_g)*(1/n_clusters_g)

train_clients, test_clients, centroids_clients, train_central, x_test_central, y_test_central = gen_data(
    perc_train, G, clusters_per_client=cpc)
test_central = {'x':x_test_central, 'y':y_test_central}

Theta_init = initialize_clients(train_clients,cpc)
Theta_fedgem_init, Pi_fedgem_init, recovered_cpc = preprocess_clients(train_clients)
true_cluster_number = true_unique_clusters(centroids_clients)

### AFCL
param_AFCL = {'k': int(np.sum(cpc)), 'max_iter': 20}
init_AFCL = {'init_seeds': Theta_init}

AFCL_clustering = AFCL(param_AFCL,init_AFCL)

start_AFCL = timeit.default_timer()
M,Q = AFCL_clustering.train(train_clients)
stop_AFCL = timeit.default_timer()
runtime_AFCL = stop_AFCL - start_AFCL

ar_AFCL, sil_AFCL= AFCL_clustering.test(test_central)

pred_vec = np.zeros(len(train_central))

for i in range(len(train_central)):
    dist_vec = np.zeros(int(np.sum(cpc)))
    for r in range(int(np.sum(cpc))):
        dist_vec[r] = np.linalg.norm(train_central[i] - M[r])
        
    pred_vec[i] = np.argmin(dist_vec)
    
n_clusters_AFCL = len(np.unique(pred_vec))

### k_FED
params_k_FED = {'K':n_clusters}
init_k_FED = {'theta': Theta_init}

k_FED_clustering = k_FED(params_k_FED,init_k_FED)

start_k_FED = timeit.default_timer()
centroids_k_FED = k_FED_clustering.train(train_clients)
stop_k_FED = timeit.default_timer()

runtime_k_FED = stop_k_FED - start_k_FED

ar_k_FED, sil_k_FED = k_FED_clustering.test(test_central)

### FedKmeans
params_FedKmeans = {'K': n_clusters, 'G': G, 'max_iter': 20}

FedKmeans_clustering = FedKmeans(params_FedKmeans)

start_FedKmeans = timeit.default_timer()
centroids_FedKmeans = FedKmeans_clustering.train(train_clients)
stop_FedKmeans = timeit.default_timer()

runtime_FedKmeans = stop_FedKmeans - start_FedKmeans

ar_FedKmeans, sil_FedKmeans = FedKmeans_clustering.test(test_central)

### FFCM_avg1
params_FFCM1 = {'K': n_clusters, 'G': G, 'P': n_features, 'max_iter': 20}

FFCM1_clustering = FFCM_avg1(params_FFCM1)

start_FFCM1 = timeit.default_timer()
centroids_FFCM1 = FFCM1_clustering.train(train_clients)
stop_FFCM1 = timeit.default_timer()

runtime_FFCM1 = stop_FFCM1 - start_FFCM1

ar_FFCM1, sil_FFCM1 = FFCM1_clustering.test(test_central)

### FFCM_avg2
params_FFCM2 = {'K': n_clusters, 'G': G, 'P': n_features, 'max_iter': 20}

FFCM2_clustering = FFCM_avg2(params_FFCM2)

start_FFCM2 = timeit.default_timer()
centroids_FFCM2 = FFCM2_clustering.train(train_clients)
stop_FFCM2 = timeit.default_timer()

runtime_FFCM2 = stop_FFCM2 - start_FFCM2

ar_FFCM2, sil_FFCM2 = FFCM2_clustering.test(test_central)

### Central GMM
central_GMM = GaussianMixture(n_components=n_clusters,
                            covariance_type='spherical',
                            init_params='k-means++')

start_GMM = timeit.default_timer()
central_GMM.fit(train_central)
stop_GMM = timeit.default_timer()

runtime_GMM = stop_GMM - start_GMM

y_pred_GMM = central_GMM.predict(x_test_central)

ar_GMM = adjusted_rand_score(y_test_central,y_pred_GMM)
sil_GMM = silhouette_score(x_test_central,y_pred_GMM)

### Central DpGMM
central_DpGMM = BayesianGaussianMixture(n_components=int(np.sum(cpc)),
                                        covariance_type='spherical',
                                        init_params='k-means++')

start_DpGMM = timeit.default_timer()
central_DpGMM.fit(train_central)
stop_DpGMM = timeit.default_timer()

runtime_DpGMM = stop_DpGMM - start_DpGMM

y_pred_DpGMM = central_DpGMM.predict(x_test_central)

ar_DpGMM = adjusted_rand_score(y_test_central,y_pred_DpGMM)
sil_DpGMM = silhouette_score(x_test_central,y_pred_DpGMM)

y_pred_train = central_DpGMM.predict(train_central)

n_clusters_DpGMM = len(np.unique(y_pred_train))


### Ours     

param_clem = {'rad':2e0, 'local_steps':1, 't_steps':10, 'max_iter':10, 'aggregate_final':False, 'P':n_features}
init_clem = {'theta_dict':Theta_fedgem_init, 'pi_dict':Pi_fedgem_init}

clem_clustering = FedClem(param_clem,init_clem)

start_clem = timeit.default_timer()
final_centroids, final_clusters = clem_clustering.train(train_clients)
stop_clem = timeit.default_timer()

runtime_clem = stop_clem - start_clem

ar_clem_vec, sil_clem_vec = clem_clustering.test(test_clients)

ar_clem = np.sum(ar_clem_vec*client_weights)
sil_clem = np.sum(sil_clem_vec*client_weights)


n_clusters_clem = len(final_clusters)

exp_results = {'ar_AFCL':ar_AFCL,
                'sil_AFCL':sil_AFCL,
                'n_clusters_AFCL':n_clusters_AFCL,
                'runtime_AFCL': runtime_AFCL,

                'ar_k_FED':ar_k_FED,
                'sil_k_FED':sil_k_FED,
                'runtime_k_FED': runtime_k_FED,

                'ar_FFCM1':ar_FFCM1,
                'sil_FFCM1':sil_FFCM1,
                'runtime_FFCM1': runtime_FFCM1,

                'ar_FFCM2':ar_FFCM2,
                'sil_FFCM2':sil_FFCM2,
                'runtime_FFCM2': runtime_FFCM2,

                'ar_FedKmeans':ar_FedKmeans,
                'sil_FedKmeans':sil_FedKmeans,
                'runtime_FedKmeans': runtime_FedKmeans,

                'ar_GMM':ar_GMM,
                'sil_GMM':sil_GMM,
                'runtime_GMM': runtime_GMM,

                'ar_DpGMM':ar_DpGMM,
                'sil_DpGMM':sil_DpGMM,
                'n_clusters_DpGMM':n_clusters_DpGMM,
                'runtime_DpGMM': runtime_DpGMM,


                'ar_clem':ar_clem,
                'sil_clem':sil_clem,
                'n_clusters_clem':n_clusters_clem,
                'runtime_clem': runtime_clem,
                'recovered_cpc': recovered_cpc,
                }

filename = 'FedClem_MNIST_'+str(int(timeit.default_timer()))+str(np.random.randint(0,high=1000))+'.mat'
scipy.io.savemat(filename,exp_results)
