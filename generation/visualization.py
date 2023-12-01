import numpy as np
import matplotlib.pyplot as plt
import json
import networkx as nx
import random

def visualization(G):
    labels = nx.get_node_attributes(G, 'label')
    embeddings = nx.get_node_attributes(G, 'embedding')

    target_node = 0

    # Get neighbors of the target_node (nodes directly connected to node 0)
    connected_nodes = list(G.neighbors(target_node)) + [target_node]

    # Create subgraph from the connected nodes
    subgraph = G.subgraph(connected_nodes)

    # Visualize the subgraph
    pos = nx.spring_layout(subgraph)
    node_colors = [labels[node] for node in subgraph.nodes()]

    nx.draw(subgraph, pos, with_labels=True, node_size=500, node_color=node_colors)
    plt.savefig("subgraph_visualization_with_color.png")
    plt.show()