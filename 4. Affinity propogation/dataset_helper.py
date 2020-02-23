import os
import numpy as np
import scipy.sparse as sp
import networkx as nx
from sklearn.externals import joblib

number_of_nodes = 196591
number_of_edges = 950327
dataset_folder_name = 'Dataset'
edges_file_name = 'loc-gowalla_edges.txt'
csr_edges_file_name = 'csr_edges.npz'
graph_file_name = 'graph.dump'


def create_graph():
    edges_file_path = os.path.join(dataset_folder_name, edges_file_name)
    edges_txt = np.loadtxt(edges_file_path)

    graph = nx.Graph()
    for edge_txt in edges_txt:
        graph.add_edge(int(edge_txt[0]), int(edge_txt[1]))

    return graph


def dump_graph():
    graph = create_graph()
    graph_file_path = os.path.join(dataset_folder_name, graph_file_name)
    joblib.dump(graph, graph_file_path)


def restore_graph():
    graph_file_path = os.path.join(dataset_folder_name, graph_file_name)
    return joblib.load(graph_file_path)


def create_edges_csr():
    edges_file_path = os.path.join(dataset_folder_name, edges_file_name)
    edges_txt = np.loadtxt(edges_file_path)

    edges_lil = sp.lil_matrix((number_of_nodes, number_of_nodes))
    for edge_txt in edges_txt:
        edges_lil[edge_txt[0], edge_txt[1]] = 1

    edges_csr = edges_lil.tocsr()
    return edges_csr


def dump_edges_csr():
    edges_csr = create_edges_csr()
    csr_edges_file_path = os.path.join(dataset_folder_name, csr_edges_file_name)
    sp.save_npz(csr_edges_file_path, edges_csr)


def restore_edges_csr():
    csr_edges_file_path = os.path.join(dataset_folder_name, csr_edges_file_name)
    return sp.load_npz(csr_edges_file_path)


a = np.array([-2] * 3)
x = 3