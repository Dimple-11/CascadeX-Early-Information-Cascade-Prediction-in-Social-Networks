import numpy as np
import networkx as nx

def common_neighbors_score(G, u, v):
    """Calculates common neighbors count between u and v (undirected projection)."""
    if not G.has_node(u) or not G.has_node(v):
        return 0
    u_neighbors = set(G.neighbors(u))
    v_neighbors = set(G.neighbors(v))
    return len(u_neighbors.intersection(v_neighbors))

def jaccard_coefficient_score(G, u, v):
    """Calculates Jaccard Coefficient score between u and v."""
    if not G.has_node(u) or not G.has_node(v):
        return 0.0
    u_neighbors = set(G.neighbors(u))
    v_neighbors = set(G.neighbors(v))
    union_len = len(u_neighbors.union(v_neighbors))
    if union_len == 0:
        return 0.0
    return len(u_neighbors.intersection(v_neighbors)) / union_len

def adamic_adar_score(G, u, v):
    """Calculates Adamic-Adar Index score between u and v."""
    if not G.has_node(u) or not G.has_node(v):
        return 0.0
    u_neighbors = set(G.neighbors(u))
    v_neighbors = set(G.neighbors(v))
    common = u_neighbors.intersection(v_neighbors)
    score = 0.0
    for w in common:
        deg = G.degree(w)
        if deg > 1:
            score += 1.0 / np.log(deg)
    return score

def preferential_attachment_score(G, u, v):
    """Calculates Preferential Attachment score between u and v."""
    if not G.has_node(u) or not G.has_node(v):
        return 0
    return G.degree(u) * G.degree(v)

def resource_allocation_score(G, u, v):
    """Calculates Resource Allocation score between u and v."""
    if not G.has_node(u) or not G.has_node(v):
        return 0.0
    u_neighbors = set(G.neighbors(u))
    v_neighbors = set(G.neighbors(v))
    common = u_neighbors.intersection(v_neighbors)
    score = 0.0
    for w in common:
        deg = G.degree(w)
        if deg > 0:
            score += 1.0 / deg
    return score
