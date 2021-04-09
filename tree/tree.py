import functools
import json
from tree import graph, nlp
import networkx as nx
import pandas as pd
import matplotlib.cm as cm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


@pd.api.extensions.register_dataframe_accessor("tree")
class TreeAccessor:
    def __init__(self, pandas_obj, compute_edges_with=None):
        self._validate(pandas_obj)
        self._obj = pandas_obj
        if not compute_edges_with:
            self._compute_edges = default_compute_edges
        self._preprocess()
        self._process()

    @staticmethod
    def _validate(obj):
        # verify there is a column text
        if "text" not in obj.columns:
            raise AttributeError("Must have 'text' column.")

    def _preprocess(self):
        # verify there is a column text
        columns_to_have = ['cluster_size', 'parent_cluster_id', 'path']

        if not set(columns_to_have).issubset(set(self._obj.columns)):
            self._obj['cluster_size'] = len(list(self._obj['text']))
            self._obj['parent_cluster_id'] = -1
            self._obj['cluster_id'] = 0
            self._obj['path'] = '0'

        if 'id' not in self._obj.columns:
            self._obj['id'] = self._obj.index + 1

    def _process(self):
        self._steps_df = []
        prev_leaves = []
        lvl = 0
        current_leaves = self.leaves
        while prev_leaves != current_leaves:
            lvl += 1
            for cluster in current_leaves:
                cluster_df = self.sub_df(cluster)
                G = generate_graph(cluster_df, compute_edges=self._compute_edges, lvl=lvl)
                subgraph_df = cluster_graph(G)

                self.update_graph_df(subgraph_df)

            self._steps_df.append(self._obj.copy())
            prev_leaves = current_leaves[:]
            current_leaves = self.leaves

    def paths(self, cid=None):
        """ Return all paths containing cid, all paths if cid is None """

        apaths = []
        paths = [path.split(' > ') for path in list(self._obj['path'].unique())]
        for path in paths:
            path = [int(x) for x in path]
            add_path = True
            if cid is not None:
                add_path = cid in path
            if add_path:
                apaths.append(path)

        return apaths

    @property
    def clusters(self):
        """ Return all clusters """

        clusters = []
        for path in self.paths():
            clusters += path

        clusters = sorted(list(set(clusters)))
        return clusters

    @property
    def nb_levels(self):
        return len(functools.reduce(lambda pa, pb: pa if len(pa) > len(pb) else pb, self.paths(), []))

    def parent(self, cid):
        path = self.paths(cid)[0]
        index = path.index(cid)
        if index:
            return path[index - 1]

        return None

    def level(self, cid):
        path = self.paths(cid)[0]
        return path.index(cid) + 1

    def siblings(self, cid):
        parent = self.parent(cid)
        parent_index = self.level(parent) - 1

        parent_paths = self.paths(parent)

        siblings = []
        for path in parent_paths:
            siblings.append(path[parent_index + 1])

        siblings = sorted(list(set(siblings)))
        return siblings

    def children(self, cid):
        index = self.level(cid) - 1

        paths = self.paths(cid)

        children = []
        for path in paths:
            if len(path) > index + 1:
                children.append(path[index + 1])

        children = sorted(list(set(children)))
        return children

    def size(self, cid):
        direct_cluster_sizes = self._obj.loc[self._obj['cluster_id'] == cid]['cluster_size'].unique()
        if direct_cluster_sizes.any():
            return max(direct_cluster_sizes)

        children = self.children(cid)
        return sum([self.size(child_id) for child_id in children])

    def sub_df(self, cid, floating_only=False):
        sub_df = self._obj.loc[self._obj['cluster_id'] == cid]
        if not floating_only:
            for child in self.children(cid):
                child_sub_df = self.sub_df(child, floating_only)
                sub_df = sub_df.append(child_sub_df)
        return sub_df

    def keywords(self, cid):
        # word_and_score = nlp.extract_keywords(corpus, nb_keywords=nlp.word_count(corpus))
        # return [word for word in word_and_score.keys()]
        # return str(self.sub_df(self, cid)['target'].value_counts(normalize=True).
        # mul(100).round(1).astype(str) + '%').split('\n')[:-1]
        corpus = ';'.join(list(self.sub_df(self, cid)['text']))
        return nlp.google_extract_keywords(corpus=corpus)

    def to_dict(self, cid):
        node = {
            # 'name': '\n'.join(self.keywords(cid)),
            'ID': int(cid),
            'size': int(self.size(cid))
        }

        children = self.children(cid)
        if children:
            node['children'] = [self.to_dict(child) for child in children]
        return node

    def to_json_file(self, cid, filename):
        with open(filename, 'w') as fp:
            json.dump(self.to_dict(cid), fp)

    def update_graph_df(self, subgraph_df):
        latest_cluster_id = max(self._obj['cluster_id'])
        subgraph_df = subgraph_df[subgraph_df['cluster_size'] > 1]
        for i, row in subgraph_df.iterrows():
            pcid = self._obj.loc[self._obj['id'] == row['id']]['cluster_id'].iloc[0]
            cid = row['cluster_id'] + latest_cluster_id + 1
            csize = row['cluster_size']
            self._obj.loc[self._obj['id'] == row['id'], 'parent_cluster_id'] = pcid
            self._obj.loc[self._obj['id'] == row['id'], 'cluster_id'] = cid
            self._obj.loc[self._obj['id'] == row['id'], 'cluster_size'] = csize
            self._obj.loc[self._obj['id'] == row['id'], 'path'] += f' > {cid}'
        return self._obj

    @property
    def leaves(self):
        clusters = list(self._obj['cluster_id'])
        parent_clusters = list(self._obj['parent_cluster_id'])
        return list(set(clusters) - set(parent_clusters))

    def plot(self, step=None):
        df = self._steps_df[step - 1] if step is not None else self._steps_df[-1]
        # Generate tree graph
        G = nx.Graph()

        # Add edges
        edges = []

        paths = [path.split(' > ') for path in list(df['path'].unique())]

        for path in paths:
            for i in range(len(path)):
                try:
                    edge = (path[i], path[i + 1])
                    if path[i] != -1 and edge not in edges:
                        edges.append(edge)
                except:
                    pass

        clusters = [int(cid) for cid in list(set(list(sum(edges, ()))))]

        G.add_edges_from(edges)

        plt.subplots(1, 1, figsize=(10, 10))

        pos = nx.spring_layout(G)

        # color the nodes according to their partition
        cmap = cm.get_cmap('viridis', max(clusters) + 1)

        nx.draw_networkx(G, pos, cmap=cmap)


def cluster_graph(G):
    # Cluster the graph

    G = graph.cluster_graph(G, graph.ClusteringMethod.LOUVAIN.value)

    clusters = graph.get_graph_clusters(G)

    # Print the clustered graph
    questions = {cluster_id: [(node['id'], node['label']) for node in cluster_nodes] for cluster_id, cluster_nodes in
                 clusters.items()}
    rows_list = []
    for cluster_id in questions:
        for qid in range(len(questions[cluster_id])):
            rows_list.append({
                'id': questions[cluster_id][qid][0],
                'text': questions[cluster_id][qid][1],
                'cluster_id': cluster_id,
                'cluster_size': len(questions[cluster_id]),
            })

    # return the graph dataframe
    return pd.DataFrame(rows_list)


def default_compute_edges(df, lvl):
        # Generate similarity matrix
        thresholds = [0.5, 0.57434918, 0.65975396, 0.75785828, 0.87055056, 1.]

        question_ids = list(df['id'])
        questions = list(df['text'])
        embeddings = []
        p_model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
        for q in questions:
            embeddings.append(p_model.encode(q))

        sim_matrix = cosine_similarity(embeddings)

        dim = sim_matrix.shape[0]
        edges = []
        for i in range(dim):
            # kwi = nlp.google_extract_keywords(corpus=questions[i])
            for j in range(dim):
                # kwj = nlp.google_extract_keywords(corpus=questions[j])
                # common = list(set.intersection(set(kwi), set(kwj)))
                if sim_matrix[i][j] > thresholds[lvl - 1]:
                    edges.append((i + 1, j + 1))

        return edges


def generate_graph(df, compute_edges, lvl):
    G = nx.Graph()

    # Add edges
    edges = compute_edges(df, lvl)

    # Add nodes
    question_ids = list(df['id'])
    questions = list(df['text'])

    nodes = [(index + 1, {'id': question_ids[index], 'label': questions[index]}) for index in range(len(question_ids))]
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    return G
