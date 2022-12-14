import os

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

import sys
import BayesNet, BNReasoner
import networkx as nx
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(SCRIPT_DIR))

DOG_FILE = os.path.join(SCRIPT_DIR, "dog_problem.BIFXML")
LEC1_FILE = os.path.join(SCRIPT_DIR, "lecture_example.BIFXML")
LEC2_FILE = os.path.join(SCRIPT_DIR, "lecture_example2.BIFXML")

USE_CASE_FILE = os.path.join(SCRIPT_DIR, "UseCase.BIFXML")


def visualize(
    br: BNReasoner,
    show_cpts=True,
    func=print,
    node_size=3000,
    interactions=False,
    show=True,
):
    br.bn.draw_structure()
    if show:
        plt.clf()
    int_graph = br.bn.get_interaction_graph()
    nx.draw(int_graph, with_labels=True)
    if show:
        plt.show()
        plt.clf()

    if show_cpts:
        print("all cpts:")
        cpts = br.bn.get_all_cpts()
        for k, cpt in cpts.items():
            print(f"{k}")
            func(cpt)
