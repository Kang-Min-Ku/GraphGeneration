import torch
from networkx.classes.graph import Graph
import networkx as nx

def homophily_ratio(graph:Graph):
    assert "label" in graph.nodes[0].keys()

    same_label_edges = sum(1 for edge in graph.edges if graph.nodes[edge[0]]["label"] == graph.nodes[edge[1]]["label"])
    total_edges = len(graph.edges)
    homophily_ratio = same_label_edges / total_edges
    
    return homophily_ratio

