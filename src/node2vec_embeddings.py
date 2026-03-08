
from node2vec import Node2Vec


def generate_node2vec_embeddings(graph):

    node2vec = Node2Vec(
        graph,
        dimensions=300,
        walk_length=30,
        num_walks=200,
        workers=4
    )

    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    node2vec_model = model.wv

    node_embeddings = {
        node: node2vec_model[node]
        for node in graph.nodes()
        if node in node2vec_model
    }

    return node_embeddings
