
import networkx as nx
from itertools import combinations


def create_graph(df):

    G = nx.Graph()

    for terms_list in df['aspectTerms']:

        for term_dict in terms_list:

            G.add_node(term_dict['term'])

            for other_term_dict in terms_list:

                if term_dict != other_term_dict:
                    G.add_edge(term_dict['term'], other_term_dict['term'])

    return G


def create_graph_kg2(df):

    nodes = set()
    word_co_occurrence = {}

    for terms_list in df['aspectTerms']:

        terms = [term_dict['term'] for term_dict in terms_list]

        for term1, term2 in combinations(terms, 2):

            edge = tuple(sorted([term1, term2]))

            word_co_occurrence[edge] = word_co_occurrence.get(edge, 0) + 1

        nodes.update(terms)

    G = nx.Graph()
    G.add_nodes_from(nodes)

    for edge, weight in word_co_occurrence.items():
        G.add_edge(*edge, weight=weight)

    return G
