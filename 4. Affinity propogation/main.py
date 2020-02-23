import dataset_helper
import checkins_helper
import scipy.sparse as sp
import numpy as np
import pandas as pd


def compute_S_and_indices(graph, edge_value=1, cycle_edge_value=-2, add_noise=False):
    nodes = list(graph.nodes())
    number_of_nodes = len(nodes)
    S = sp.lil_matrix((number_of_nodes, number_of_nodes), dtype=np.float32)

    for node in nodes:
        neighbors_list = list(graph.neighbors(node))
        if add_noise:
            S[node, neighbors_list] = edge_value + np.random.random(len(neighbors_list)) * 1e-15
        else:
            S[node, neighbors_list] = edge_value

    S.setdiag(cycle_edge_value)
    S_keys = sp.find(S)
    S_keys_rows = S_keys[0].reshape(1, -1)
    S_keys_columns = S_keys[1].reshape(1, -1)
    S_indices = np.concatenate((S_keys_rows, S_keys_columns), axis=0)
    return S, S_indices


def affinity_propogation(S, S_indices, max_iterations=15, smoothing_factor=0.5):
    csr_correction_value = 10000.
    number_of_samples = S.shape[0]

    A = sp.lil_matrix((number_of_samples, number_of_samples), dtype=np.float32)
    R = sp.lil_matrix((number_of_samples, number_of_samples), dtype=np.float32)
    S_csr = S.tocsr()
    rows = np.arange(0, number_of_samples)
    
    csr_correction = sp.lil_matrix((number_of_samples, number_of_samples), dtype=np.float32)
    csr_correction[S_indices[0], S_indices[1]] = csr_correction_value
    csr_correction = csr_correction.tocsr()
    
    for iteration in range(max_iterations):
        # Calculate responsibilities
        tmp_sums = A.tocsr() + S_csr + csr_correction
        tmp_sums.eliminate_zeros()

        # Find first maximum arguments
        rows_maximum_args = np.asarray(tmp_sums.argmax(axis=1)).flatten()

        # Find second maximums
        tmp_sums_copy = tmp_sums.copy()
        tmp_sums_copy[rows, rows_maximum_args] = -np.inf
        rows_pre_maximum_args = np.asarray(tmp_sums_copy.argmax(axis=1)).flatten()

        # Prepare maximums matrix
        tmp_maximums = sp.lil_matrix((number_of_samples, number_of_samples), dtype=np.float32)
        tmp_maximums[S_indices[0], S_indices[1]] = np.asarray(tmp_sums[rows, rows_maximum_args]).flatten()[S_indices[0]]
        tmp_maximums[rows, rows_maximum_args] = tmp_sums[rows, rows_pre_maximum_args]

        maximums = sp.lil_matrix((number_of_samples, number_of_samples), dtype=np.float32)
        maximums[S_indices[0], S_indices[1]] = tmp_maximums[S_indices[0], S_indices[1]]
        maximums = maximums.tocsr()
        maximums -= csr_correction
        maximums.eliminate_zeros()

        old_R = R.copy()
        R = S_csr - maximums
        R.eliminate_zeros()
        
        # Exponential smoothing
        R = smoothing_factor * old_R + (1. - smoothing_factor) * R

        # Calculate availabilities        
        # Remove all values which are less then zero
        tmp_R = R.copy()
        tmp_R[tmp_R < 0] = 0
        tmp_R = tmp_R.tocsr()
        tmp_R.eliminate_zeros()
        
        tmp_R_diagonal = tmp_R.diagonal()   
        tmp_R_column_sums_minus_diagonal = np.asarray(tmp_R.sum(axis=0)).flatten() \
                                           - tmp_R_diagonal
                            
        tmp_A = sp.lil_matrix((number_of_samples, number_of_samples), dtype=np.float32)
        tmp_A[S_indices[0], S_indices[1]] = (tmp_R_column_sums_minus_diagonal + R.diagonal()).flatten()[S_indices[1]]
        tmp_A[S_indices[0], S_indices[1]] -= tmp_R[S_indices[0], S_indices[1]]
        
        old_A = A.copy()
        tmp_A[tmp_A > 0] = 0
        tmp_A.setdiag(tmp_R_column_sums_minus_diagonal)        
        A = tmp_A.tocsr()
        A.eliminate_zeros()
        
        # Exponential smoothing
        A = smoothing_factor * old_A + (1. - smoothing_factor) * A
        
        number_of_clusters = len(np.unique(get_cluster_indexes(A, R, S_indices)))
        print ('Iteration: ' + str(iteration) + '\tNumber of clusters: ' + str(number_of_clusters))
    return A, R


def calculate_cluster_sizes(cluster_indexes):
    number_of_samples = len(cluster_indexes)
    clusters = np.ndarray((number_of_samples,), dtype=np.ulonglong)
    
    for cluster_index in cluster_indexes:
        clusters[cluster_index] += 1
        
    return clusters


def get_cluster_indexes(A, R, S_indices, transitive_convolution_depth=1000):
    number_of_samples = A.shape[0]
    
    tmp = sp.lil_matrix((number_of_samples, number_of_samples), dtype=np.float32)
    tmp[S_indices[0], S_indices[1]] = A[S_indices[0], S_indices[1]] + R[S_indices[0], S_indices[1]] + 1.
    tmp = tmp.tocsr()
    
    cluster_indexes = np.asarray(tmp.argmax(axis=1)).flatten()
    
    for i in range(transitive_convolution_depth):
        cluster_indexes[:] = cluster_indexes[cluster_indexes[:]]
    
    return cluster_indexes


def get_clusters_df(cluster_indexes):
    cluster_centers, clusters_count = np.unique(cluster_indexes, return_counts=True)
    clusters = np.concatenate((cluster_centers.reshape(-1, 1),
                               clusters_count.reshape(-1, 1)), axis=1)
    clusters_df = pd.DataFrame(clusters)
    clusters_df.columns = ['Cluster index', 'Count']
    clusters_df = clusters_df.sort_values(by='Count', ascending=False)
    clusters_df.reset_index(drop=True, inplace=True)
    return clusters_df


def get_combined_df(cluster_indexes, checkins):
    checkins_df = pd.DataFrame(checkins, dtype=np.int32)
    checkins_df.columns = ['UserId', 'LocationId']
    
    clusters_df = pd.DataFrame(cluster_indexes, dtype=np.int32)
    clusters_df = clusters_df.reset_index()
    clusters_df.columns = ['UserId', 'ClusterId']
    
    return pd.merge(checkins_df, clusters_df, on='UserId')


def split_for_test_and_train_dfs(users, combined_df):
    number_of_users = users.size
    test_size = number_of_users // 10

    permutation = np.random.permutation(number_of_users)
    users_random = users[permutation]

    test_users = users_random[:test_size]
    train_users = users_random[test_size:]

    test_df = combined_df.loc[combined_df.UserId.isin(test_users)]
    train_df = combined_df.loc[combined_df.UserId.isin(train_users)]
    return test_users, test_df, train_df


def get_top_k_locations(train_df, k):
    locations = train_df.groupby('LocationId')
    locations_count = locations.count()
    locations_count = locations_count.drop('ClusterId', axis=1)
    locations_count = locations_count.rename(columns={'UserId': 'Count'})

    locations_ranked = locations_count.sort_values('Count', ascending=False)
    locations_ranked = locations_ranked.reset_index()
    locations_top_k_df = locations_ranked[:k]
    locations_top_k = locations_top_k_df.LocationId.values
    locations_top_k_uniq = np.unique(locations_top_k)
    return locations_top_k, locations_top_k_uniq


def rank_cluster_locations(train_df):
    cluster_locations = train_df.groupby(['ClusterId', 'LocationId'])
    cluster_locations_count = cluster_locations.count()
    cluster_locations_count = cluster_locations_count.rename(columns={'UserId': 'Count'})

    cluster_locations_ranked = cluster_locations_count.sort_values(['ClusterId', 'Count'], ascending=False)
    return cluster_locations_ranked.reset_index()


def calculate_metrics(cluster_indexes, test_users, test_df, locations_top_k_uniq, cluster_locations_ranked, k):
    location_intersections = 0
    cluster_hits = 0    
    
    for user in test_users:
        all_locations = test_df.loc[test_df.UserId == user]['LocationId'].values
        locations_user_uniq = np.unique(all_locations)
    
        location_intersections += np.intersect1d(locations_top_k_uniq, locations_user_uniq).size
    
        cluster_index = cluster_indexes[user]
        cluster_locations_top_k = cluster_locations_ranked.loc[cluster_locations_ranked.ClusterId==cluster_index]['LocationId'].values[:k]
        cluster_locations_top_k_uniq = np.unique(cluster_locations_top_k)
        cluster_hits += np.intersect1d(cluster_locations_top_k_uniq, locations_user_uniq).size

    test_size = test_users.size
    location_prediction_acc = location_intersections / (k * test_size)
    cluster_acc = cluster_hits / (k * test_size)
    return location_prediction_acc, cluster_acc


k = 10
print("Restoring dataset... ", end='')
graph = dataset_helper.restore_graph()
print("Success")

print("Computing S matrix... ", end='')
S, S_indices = compute_S_and_indices(graph, edge_value=1, cycle_edge_value=-1.5, add_noise=True)
print("Success")

print("Processing affinity propogation...")
A, R = affinity_propogation(S, S_indices, max_iterations=15, smoothing_factor=0.5)
print("Success")

cluster_indexes = get_cluster_indexes(A, R, S_indices)
cluster_sizes = calculate_cluster_sizes(cluster_indexes)
clusters_df = get_clusters_df(cluster_indexes)
checkins = checkins_helper.get_checkins()
users = checkins_helper.get_users(checkins)
combined_df = get_combined_df(cluster_indexes, checkins)
test_users, test_df, train_df = split_for_test_and_train_dfs(users, combined_df)
locations_top_k, locations_top_k_uniq = get_top_k_locations(train_df, k)
cluster_locations_ranked = rank_cluster_locations(train_df)

location_prediction_acc, cluster_acc = calculate_metrics(cluster_indexes, test_users, test_df, locations_top_k_uniq, cluster_locations_ranked, k)
print("Location prediction accuracy: %.6f" % location_prediction_acc)
print("Cluster accuracy: %.6f" % cluster_acc)
