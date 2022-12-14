from typing import Union, MutableSet, Dict, List, Any, Tuple
import networkx as nx
import pandas as pd
from copy import deepcopy
from enum import Enum

import os


SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

import sys

sys.path.append(os.path.dirname(SCRIPT_DIR))
from BayesNet import BayesNet
from BNReasoner import BNReasoner, Ordering
from examples import USE_CASE_FILE, visualize
import random_network


# bn = BayesNet()
# bn.load_from_bifxml('examples/UseCase.BIFXML')
# bn.draw_structure()
# var = bn.get_all_variables()
#
# cpts = bn.get_all_cpts()


def cpt_latex():
    bn = BayesNet()
    bn.load_from_bifxml(USE_CASE_FILE)
    var = bn.get_all_variables()

    for element in var:
        cpt = bn.get_cpt(element)
        print(cpt.to_latex(index=False))


# print(type(cpts))


def main():
    # cpt_latex()
    br = BNReasoner(USE_CASE_FILE)
    # visualize(br, node_size=500)

    # interesting_queries()
    rand_network()


"""
What’s the effect of the min degree vs min fill heuristic
on the performance of variable elimination?

# generated the below using (then fixed duplicates):
# options = [str(n) for n in range(10)]
# random.choices(options, k=4)
# (lists of vars to eliminate)
"""
ELIMINATION_QUERIES = [
    ["6", "5", "0", "1"],
    ["8", "6", "3", "2"],
    ["4", "7", "8", "5"],
    ["0", "8", "2", "6"],
    ["6", "2", "7", "9"],
]

"""
testing networking pruning on marginal dist queries.
"""

# MARGINAL_QUERIES = [
#    {
#        "Q": set(["0", "1"]),
#        "E": {"2": True, "3": False, "4": True},
#        "ordering": Ordering.MIN_DEG,
#    },
#    # 2
#    {
#        "Q": set(["0", "1"]),
#        "E": {"2": True, "3": False, "4": True},
#        "ordering": Ordering.MIN_FILL,
#    },
# ]

"""
import random
vars = [str(n) for n in range(10)]
#['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def get_one():
    Q = random.sample(vars, k=1)
    re_vars = set(vars) - set(Q)
    E_vars = random.sample(re_vars, k=2)
    E = {v: bool(random.getrandbits(1)) for v in E_vars}
    return {"Q": Q, "E": E}

MARGINAL_QUERIES = [random.choices(options, k=4) for _ in range(5)]
"""
# all of these will be run with the Ordering suggested by the results of the ELIIMINATION_QUERIES experiment
MARGINAL_QUERIES = [
    {"Q": ["4"], "E": {"8": False, "6": False}},
    {"Q": ["2"], "E": {"8": True, "4": False}},
    {"Q": ["9"], "E": {"5": False, "2": True}},
    {"Q": ["4"], "E": {"5": True, "2": False}},
    {"Q": ["7"], "E": {"2": False, "0": True}},
]


def rand_network(func=print):
    options = {
        # "n_nodes": 10,
        # "edge_prob": 0.1,
        "n_nodes": 5,
        "edge_prob": 0.5,
    }

    fname = "/tmp/cur.bifxml"
    import pdb

    pdb.set_trace()
    br = random_network.get_random_br(fname, options)
    visualize(br)


def interesting_queries(func=print):
    br = BNReasoner(USE_CASE_FILE)
    # visualize(br, node_size=500)

    res = br.marginal_distribution(
        {"arriving-on-time"}, pd.Series({"bad-weather": True})
    )
    print("marginal_dist:")
    func(res)

    # print('MAP')


if __name__ == "__main__":
    main()
