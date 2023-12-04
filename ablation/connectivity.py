import numpy as np
import networkx as nx
from networkx.classes.graph import Graph

def class_connectivity(graph:Graph, normalize=True):
    assert "label" in graph.nodes[0].keys()

    classes = [graph.nodes[node]["label"] for node in graph.nodes]
    label = set(classes)
    connectivity = np.zeros((len(label), len(label)))

    for edge in graph.edges:
        i = graph.nodes[edge[0]]["label"]
        j = graph.nodes[edge[1]]["label"]
        connectivity[i][j] += 1

    if normalize:
        connectivity = connectivity / connectivity.sum(axis=1, keepdims=True)
    
    return connectivity