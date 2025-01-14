"""
Creates random bayesian networks and outputs them to a bifxml file.
"""
import os
import sys

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
from pgmpy.base import DAG
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork
from pgmpy.readwrite import XMLBIFWriter, XMLBIFReader
import networkx as nx

import matplotlib.pyplot as plt
from typing import Dict, Union

from BayesNet import BayesNet
from BNReasoner import BNReasoner, Ordering
from examples import visualize


def get_random_model(
    n_nodes: int = 5,
    edge_prob: float = 0.5,
    n_states: Union[int, Dict, None] = 2,
    latents: bool = False,
):
    """
    Based on https://github.com/pgmpy/pgmpy/blob/dev/pgmpy/models/BayesianNetwork.py#L1063
    """
    if n_states is None:
        n_states = np.random.randint(low=1, high=5, size=n_nodes)
    elif isinstance(n_states, int):
        n_states = np.array([n_states] * n_nodes)
    else:
        n_states = np.array(n_states)

    # n_states_dict = {i: n_states[i] for i in range(n_nodes)}
    n_states_dict = {str(i): n_states[i] for i in range(n_nodes)}

    # dag = DAG.get_random(n_nodes=n_nodes, edge_prob=edge_prob, latents=latents)
    dag = DAG_get_random(n_nodes=n_nodes, edge_prob=edge_prob, latents=latents)
    bn_model = BayesianNetwork(dag.edges(), latents=dag.latents)
    bn_model.add_nodes_from(dag.nodes())

    cpds = []
    for node in bn_model.nodes():
        parents = list(bn_model.predecessors(node))
        cpds.append(
            TabularCPD.get_random(
                variable=node, evidence=parents, cardinality=n_states_dict
            )
        )

    bn_model.add_cpds(*cpds)
    return bn_model


def DAG_get_random(n_nodes=5, edge_prob=0.5, latents=False):
    """
    Based on https://github.com/pgmpy/pgmpy/blob/dev/pgmpy/base/DAG.py#L1016
    """
    # Step 1: Generate a matrix of 0 and 1. Prob of choosing 1 = edge_prob
    adj_mat = np.random.choice(
        [0, 1], size=(n_nodes, n_nodes), p=[1 - edge_prob, edge_prob]
    )

    # Step 2: Use the upper triangular part of the matrix as adjacency.
    # nodes = list(range(n_nodes))
    nodes = [str(n) for n in range(n_nodes)]
    mat = np.triu(adj_mat, k=1)

    graph = nx.convert_matrix.from_numpy_matrix(mat, create_using=nx.DiGraph)
    # magic sauce:
    #   https://networkx.org/documentation/stable/reference/generated/networkx.relabel.relabel_nodes.html
    graph = nx.relabel_nodes(graph, {n: str(n) for n in graph.nodes()})
    edges = graph.edges()

    dag = DAG(edges)
    dag.add_nodes_from(nodes)
    if latents:
        dag.latents = set(
            np.random.choice(
                dag.nodes(), np.random.randint(low=0, high=len(dag.nodes()))
            )
        )
    return dag


def get_random_br(fname: str, args: Dict) -> BNReasoner:
    model = get_random_model(**args)
    XMLBIFWriter(model).write_xmlbif(fname)
    print(f"wrote: '{fname}'")
    assert fname.endswith(".bifxml")
    assert os.path.exists(fname)
    return BNReasoner(fname)


if __name__ == "__main__":
    # br = BNReasoner("random.bifxml")
    # visualize(br)

    for i in range(10):
        fname = f"random{i}.bifxml"
        br = get_random_br(fname, {"n_nodes": 10, "edge_prob": 0.1, "n_states": 2})
        visualize(br)

    exit(0)

    # model = BayesianNetwork.get_random(n_nodes=5)

    model = get_random_model(n_nodes=20, edge_prob=0.5, n_states=2)
    XMLBIFWriter(model).write_xmlbif(fname)
    print(f"wrote {fname}")

    # plt.title("random graph")
    # nx.draw(model, with_labels=True)
    # plt.savefig("random.png")

    # https://pgmpy.org/readwrite/xmlbif.html#pgmpy.readwrite.XMLBIF.XMLBIFReader.get_model
    model2 = XMLBIFReader(fname).get_model()
    # plt.clf()
    # plt.title("random graph (stage 2)")
    # nx.draw(model2, with_labels=True)
    # plt.savefig("random-stage2.png")

    # test loading model into our class
    # bn = BayesNet()
    # bn.load_from_bifxml(fname)
    br = BNReasoner(fname)
    visualize(br)
