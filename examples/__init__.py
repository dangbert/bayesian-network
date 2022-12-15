import os

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

import sys
from BayesNet import BayesNet
from BNReasoner import BNReasoner
import networkx as nx
import matplotlib.pyplot as plt
from typing import Union

sys.path.append(os.path.dirname(SCRIPT_DIR))

DOG_FILE = os.path.join(SCRIPT_DIR, "dog_problem.BIFXML")
LEC1_FILE = os.path.join(SCRIPT_DIR, "lecture_example.BIFXML")
LEC2_FILE = os.path.join(SCRIPT_DIR, "lecture_example2.BIFXML")

USE_CASE_FILE = os.path.join(SCRIPT_DIR, "UseCase.BIFXML")


def visualize(
    net: Union[BayesNet, BNReasoner],
    show_cpts=True,
    func=print,
    node_size=3000,
    interactions=False,
    show=True,
):
    if type(net) == BayesNet:
        bn = net
    elif type(net) == BNReasoner:
        bn = net.bn
    else:
        raise TypeError(f"{type(net)} not expected")

    node_size = 300
    bn.draw_structure(node_size=node_size)
    if show:
        plt.clf()
    if interactions:
        int_graph = bn.get_interaction_graph()
        nx.draw(int_graph, with_labels=True, node_size=node_size)
        if show:
            plt.show()
            plt.clf()

    if show_cpts:
        print("all cpts:")
        cpts = bn.get_all_cpts()
        for k, cpt in cpts.items():
            print(f"{k}")
            func(cpt)
