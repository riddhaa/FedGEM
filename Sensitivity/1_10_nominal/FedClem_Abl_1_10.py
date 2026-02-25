import gurobipy as grb
import numpy as np
import timeit
import scipy
from sklearn.datasets import make_blobs
import sklearn
from sklearn import metrics
from sklearn import model_selection
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import silhouette_score
import copy
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import pdist

def gen_data(n_train_all, n_test, n_features, n_clusters, G, client_cluster_weights, clusters_per_client=-1, min_dist=2.0):
    n_samples = int((np.sum(n_train_all) + n_test)*G*2*n_clusters)
    cluster_centers = generate_cluster_centers_unbounded(n_clusters, n_features, min_dist, max_attempts=1e10)
    
    x_all, y_all, centers = make_blobs(n_samples=n_samples, 
                                       n_features=n_features,
                                       centers=cluster_centers, 
                                       cluster_std=1.0, 
                                       shuffle=True, 
                                       random_state=None, 
                                       return_centers=True)
    
    R_max = comp_R_max(centers)
    
    data_dict = {}
    for cl in range(n_clusters):
        data_dict['x_'+str(cl)] = x_all[y_all == cl]
        data_dict['y_'+str(cl)] = y_all[y_all == cl]
    
    if any(clusters_per_client) == -1:
        clusters_per_client = np.ones(G)*n_clusters
        
    clusters_per_client = clusters_per_client.astype(int)
        
    train_clients = {}
    test_clients = {}
    centroids_clients = {}
    counters = {}
    
    for g in range(G):
        n_train = n_train_all[g]
        clusters_g = np.random.choice(n_clusters, size=clusters_per_client[g], replace=False)
        centroids_g = np.zeros([len(clusters_g),n_features])
        centroids_g[0] = centers[clusters_g[0]]
        
        weights_g = client_cluster_weights['client_'+str(g)]
        
        n_train_0 = int(weights_g[0]*n_train)
        
        x_curr = data_dict['x_'+str(clusters_g[0])]
        y_curr = data_dict['y_'+str(clusters_g[0])]
        
        x_train_g = x_curr[:n_train_0]
        x_test_g = x_curr[n_train_0:n_test]
        y_test_g = y_curr[n_train_0:n_test]
        
        x_curr = x_curr[n_train_0 + n_test:]
        y_curr = y_curr[n_train_0 + n_test:]
        
        data_dict['x_'+str(clusters_g[0])] = x_curr
        data_dict['y_'+str(clusters_g[0])] = y_curr
        
        for c in range(1,len(clusters_g)):
            centroids_g[c] = centers[clusters_g[c]]
            
            n_train_c = int(weights_g[c]*n_train)
            
            x_curr = data_dict['x_'+str(clusters_g[c])]
            y_curr = data_dict['y_'+str(clusters_g[c])]
            
            x_train_g = np.concatenate([x_train_g,x_curr[:n_train_c]],axis=0)
            x_test_g = np.concatenate([x_test_g,x_curr[n_train_c:n_test]],axis=0)
            y_test_g = np.concatenate([y_test_g,y_curr[n_train_c:n_test]],axis=0)
            
            x_curr = x_curr[n_train_c + n_test:]
            y_curr = y_curr[n_train_c + n_test:]
            
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
        
    return train_clients, test_clients, centroids_clients, train_central, x_test_central, y_test_central, R_max
        
    
def generate_cluster_centers_unbounded(n_clusters, n_features, min_dist, tolerance=1e-6, max_attempts=10000):

    centers = []
    attempts = 0


    center1 = np.random.normal(scale=5.0, size=n_features)
    direction = np.random.normal(size=n_features)
    direction /= np.linalg.norm(direction)
    center2 = center1 + min_dist * direction

    centers.append(center1)
    centers.append(center2)


    while len(centers) < n_clusters and attempts < max_attempts:
        candidate = np.random.normal(scale=5.0, size=n_features) + np.mean(centers, axis=0)
        if all(np.linalg.norm(candidate - c) >= min_dist for c in centers):
            centers.append(candidate)
        attempts += 1

    if len(centers) < n_clusters:
        raise ValueError(f"Could not generate {n_clusters} centers after {max_attempts} attempts.")

    centers = np.array(centers)

    return centers

def comp_R_max(centroids):
    R_max = 0
    for c1 in range(len(centroids)):
        for c2 in range(len(centroids)):
            curr = np.linalg.norm(centroids[c1] - centroids[c2])
            if curr > R_max:
                R_max = curr
                
    return R_max

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
        
        model = grb.Model()
        model.setParam('OutputFlag',False)
        
        var_theta = {}
        for p in range(len(theta_1)):
            var_theta[p] = model.addVar(vtype=grb.GRB.CONTINUOUS,lb=-grb.GRB.INFINITY)
            
        model.update()
        
        model.addQConstr(grb.quicksum((theta_1[p] - var_theta[p])*(theta_1[p] - var_theta[p]) 
                                      for p in range(len(theta_1))) <= eps_1)
        
        model.addQConstr(grb.quicksum((theta_2[p] - var_theta[p])*(theta_2[p] - var_theta[p]) 
                                      for p in range(len(theta_2))) <= eps_2)
        
        obj = grb.quicksum(
            (theta_1[p] - var_theta[p])*(theta_1[p] - var_theta[p]) 
                           for p in range(len(theta_1))) + grb.quicksum(
            (theta_2[p] - var_theta[p])*(theta_2[p] - var_theta[p]) 
                                      for p in range(len(theta_2)))
        
        model.setObjective(obj,grb.GRB.MINIMIZE)
        
        model.optimize()
        
        status = model.Status
        if status != grb.GRB.OPTIMAL:
            theta_opt = np.ones(theta_1.shape)*float('inf')
        else:
            theta_opt = np.array([var_theta[p].x for p in range(len(theta_1))])
            
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
        self.server_comp_aggregation(amb_sets_all)
        Theta_prev = copy.deepcopy(self.Theta_curr)
        continue_training = True
        while continue_training and l < self.max_iter:
            amb_sets_all = self.train_all_clients(train_data)
            amb_sets_all, clusters = self.server_comp_clustering(amb_sets_all)
            self.server_comp_aggregation(amb_sets_all)
            continue_training = self.check_converged(Theta_prev)
            Theta_prev = copy.deepcopy(self.Theta_curr)
            l += 1
            
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


def save_output(R_min,n_clusters,clem_ar_vec,clem_ar,clem_sil_vec,clem_sil,
                central_ar_vec,central_ar,central_sil_vec,central_sil,
                clem_num_cl_vec,pred_num_cl,true_num_cl,err_num_cl):
    exp_info = {'R_min': R_min,
                'n_clusters': n_clusters,
                'clem_ar_vec': clem_ar_vec,
                'clem_ar': clem_ar,
                'clem_sil_vec': clem_sil_vec,
                'clem_sil': clem_sil,
                'central_ar_vec': central_ar_vec,
                'central_ar': central_ar,
                'central_sil_vec': central_sil_vec,
                'central_sil': central_sil,
                'clem_num_cl_vec': clem_num_cl_vec,
                'pred_num_cl': pred_num_cl,
                'true_num_cl': true_num_cl,
                'err_num_cl': err_num_cl
    }
    
    filename = 'FedClem_Abl_'+str(R_min)+'_'+str(n_clusters)+'_'+str(int(timeit.default_timer()))+str(np.random.randint(0,high=1000))+'.mat'
    scipy.io.savemat(filename,exp_info)
    
G = 5
n_train_all = np.ones(G)*500
client_weights = n_train_all/np.sum(n_train_all)
n_test = 5000
n_features = 4
n_clusters = 10
cpc = np.random.randint(2,n_clusters-1,G)
R_min = 1

client_cluster_weights = {}
Pi_init = {}
for g in range(G):
    n_clusters_g = cpc[g]
    client_cluster_weights['client_'+str(g)] = np.ones(n_clusters_g)*(1/n_clusters_g)
    Pi_init['Pi_'+str(g)] = np.ones(n_clusters_g)*(1/n_clusters_g)

true_cluster_number = 0
while true_cluster_number < n_clusters:
    train_clients, test_clients, centroids_clients, train_central, x_test_central, y_test_central, R_max = \
    gen_data(n_train_all, n_test, n_features, n_clusters, G, client_cluster_weights, clusters_per_client=cpc, min_dist=R_min)
    true_cluster_number = true_unique_clusters(centroids_clients)
    
num_exp = 3
rad_factor_vec = [1e-1,5e-1,1e0,5e0,1e1]
n_folds = 3

clem_ar_vec = np.zeros([num_exp])
clem_sil_vec = np.zeros([num_exp])
clem_num_cl_vec = np.zeros([num_exp])

central_ar_vec = np.zeros([num_exp])
central_sil_vec = np.zeros([num_exp])

for i in range(num_exp):
    Theta_init = initialize_clients(train_clients,cpc)
    
    clem_sil_cv_arr = np.zeros([n_folds,len(rad_factor_vec)])
    
    client_train_split, client_val_split = cv_split(train_clients,G,n_folds=5)
    
    for f in range(n_folds):
        train_clients_f = {}
        val_clients_f = {}
        
        for g in range(G):
            train_clients_f['client_'+str(g)] = client_train_split['client_'+str(g)+'_'+str(f)]
            val_clients_f['x_'+str(g)] = client_val_split['x_'+str(g)+'_'+str(f)]
            
        for r in range(len(rad_factor_vec)):
            param_clem = {'rad':rad_factor_vec[r], 'local_steps':1, 't_steps':10, 'max_iter':10, 'aggregate_final':False, 'P':n_features}
            init_clem = {'theta_dict':Theta_init, 'pi_dict':Pi_init}
            
            clem_clustering = FedClem(param_clem,init_clem)
            final_centroids, final_clusters = clem_clustering.train(train_clients_f)
            sil_scores = clem_clustering.validate(val_clients_f)
            
            clem_sil_cv_arr[f,r] = np.sum(sil_scores*client_weights)
            
    clem_sil_cv_vec = np.mean(clem_sil_cv_arr,axis=0)
    r_clem = np.argmax(clem_sil_cv_vec)
    
    param_clem = {'rad':rad_factor_vec[r_clem], 'local_steps':1, 't_steps':10, 'max_iter':10, 'aggregate_final':False, 'P':n_features}
    init_clem = {'theta_dict':Theta_init, 'pi_dict':Pi_init}
    
    clem_clustering = FedClem(param_clem,init_clem)
    final_centroids, final_clusters = clem_clustering.train(train_clients)
    ar_scores, sil_scores = clem_clustering.test(test_clients)
    
    clem_ar_vec[i] = np.sum(ar_scores*client_weights)
    clem_sil_vec[i] = np.sum(sil_scores*client_weights)
    clem_num_cl_vec[i] = len(final_clusters)
    
    central = GaussianMixture(n_components=n_clusters,
                         covariance_type='spherical',
                         init_params='k-means++')
    central.fit(train_central)
    y_pred = central.predict(x_test_central)

    central_ar_vec[i] = adjusted_rand_score(y_test_central,y_pred)
    central_sil_vec[i] = silhouette_score(x_test_central,y_pred)
    
clem_ar = np.mean(clem_ar_vec)
clem_sil = np.mean(clem_sil_vec)

central_ar = np.mean(central_ar_vec)
central_sil = np.mean(central_sil_vec)

pred_num_cl = np.mean(clem_num_cl_vec)
true_num_cl = true_cluster_number
err_num_cl = (true_num_cl - pred_num_cl)/true_num_cl
    
save_output(R_min,n_clusters,clem_ar_vec,clem_ar,clem_sil_vec,clem_sil,
                central_ar_vec,central_ar,central_sil_vec,central_sil,
                clem_num_cl_vec,pred_num_cl,true_num_cl,err_num_cl)