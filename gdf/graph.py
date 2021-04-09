import json
from enum import Enum
from functools import cmp_to_key
from math import floor
from itertools import cycle

import networkx as nx
from cdlib import algorithms

from . import nlp


class ClusteringMethod(Enum):
    LOUVAIN = 'louvain'
    FAST_GREEDY = 'fast_greedy'
    MARKOV_CLUSTER = 'markov_cluster'
    INFO_MAP = 'info_map'
    WALKTRAP = 'walktrap'


def get_graph_nodes(G):
    return [G.nodes[i+1] for i in range(len(G.nodes))]


def compute_cluster_score(cluster):
    return len(cluster)


def compare_clusters(first, second):
    return compute_cluster_score(first) < compute_cluster_score(second)


class TruncationStrategy(Enum):
    NONE = None
    MOST_DIVERSE = 'most_diverse'
    HIGHEST_SCORE = 'highest_score'


def get_truncate_func(truncation_strategy):
    if truncation_strategy == TruncationStrategy.MOST_DIVERSE:
        return nlp.k_most_diverse

    elif truncation_strategy == TruncationStrategy.HIGHEST_SCORE:
        return nlp.k_highest_scoring

    else:
        return nlp.k_first


def truncate_node_list(node_list, to_size, truncation_strategy=None):
    truncate_func = get_truncate_func(truncation_strategy=truncation_strategy)
    question_node_dict = {node['label']: node for node in node_list}
    questions = list(question_node_dict.keys())
    top_questions = truncate_func(sentences=questions, k=to_size)
    top_nodes = [question_node_dict[question] for question in top_questions]
    return top_nodes


def compute_new_cluster_sizes(G, clusters, limit):
    new_cluster_sizes = {}
    remaining = limit

    for cluster_id in clusters:
        new_cluster_size = floor(limit * len(clusters[cluster_id]) / len(G.nodes))
        new_cluster_sizes[cluster_id] = new_cluster_size
        remaining -= new_cluster_size

    if remaining > 0:
        sorted_clusters = {k: v for k, v in sorted(clusters.items(), key=cmp_to_key(compare_clusters), reverse=True)}
        pool = cycle(list(sorted_clusters.keys()))

        while remaining > 0:
            cluster_id = next(pool)
            new_cluster_sizes[cluster_id] += 1
            remaining -= 1

    return new_cluster_sizes


def get_graph_clusters(G):
    clusters = {}  # key = cluster_id, value = node list

    for node in get_graph_nodes(G):
        cluster_id = node['group']
        if cluster_id not in clusters:
            clusters[cluster_id] = [node]
        else:
            clusters[cluster_id].append(node)
    return clusters


def truncate_questions_graph(G, limit, clustering_aware=False, truncation_strategy=None):
    if clustering_aware:
        # Get clusters dict
        clusters = get_graph_clusters(G)

        # Compute new sizes
        new_cluster_sizes = compute_new_cluster_sizes(G, clusters, limit)

        # Truncate clusters
        for cluster_id in clusters:
            clusters[cluster_id] = truncate_node_list(clusters[cluster_id], new_cluster_sizes[cluster_id], truncation_strategy=truncation_strategy)

        # Update graph
        for node in get_graph_nodes(G):
            cluster_id = node['group']
            if node not in clusters[cluster_id]:
                G.remove_node(node['id'])

    else:
        top_nodes = truncate_node_list(get_graph_nodes(G), limit, truncation_strategy=truncation_strategy)
        for node in get_graph_nodes(G):
            if node not in top_nodes:
                G.remove_node(node['id'])

    return G


def cluster_graph(G, clustering):
    if clustering is not None:

        node_community_map = None

        if clustering == ClusteringMethod.LOUVAIN.value:
            node_community_map = algorithms.louvain(G).to_node_community_map()

        elif clustering == ClusteringMethod.FAST_GREEDY.value:
            node_community_map = algorithms.greedy_modularity(G).to_node_community_map()

        elif clustering == ClusteringMethod.MARKOV_CLUSTER.value:
            node_community_map = {k+1: v for k, v in algorithms.markov_clustering(G).to_node_community_map().items()}

        elif clustering == ClusteringMethod.INFO_MAP.value:
            node_community_map = algorithms.infomap(G).to_node_community_map()

        elif clustering == ClusteringMethod.WALKTRAP.value:
            node_community_map = algorithms.walktrap(G).to_node_community_map()

        # Update graph with group membership
        G = assign_clusters(G, node_community_map)

    return G


def load_graph_from_json(filename):
    with open(filename) as infile:
        json_graph = json.load(infile)
        G = nx.Graph()

        # Add nodes
        nodes = [(id + 1, json_graph['nodes'][id]) for id in range(len(json_graph['nodes']))]
        G.add_nodes_from(nodes)

        # Add edges
        edges = [(edge['from'], edge['to']) for edge in json_graph['edges']]
        G.add_edges_from(edges)

        return G


def assign_clusters(G, node_community_map):
    for i in range(len(G.nodes)):
        node_communities = node_community_map[i+1]

        if node_communities:
            G.nodes[i+1]['group'] = node_communities[0]

        else:
            G.nodes[i+1]['group'] = -1  # belong to no community

    return G


def graph_to_dict(G):
    return {
        'nodes': [G.nodes[key] for key in G.nodes.keys()],
        'edges': [{'from': edge[0], 'to':edge[1]} for edge in G.edges]
    }


def graph_to_json_file(G, filename):
    with open(filename, 'w') as outfile:
        json.dump(graph_to_dict(G), outfile)
