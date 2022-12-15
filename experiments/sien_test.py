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


# bn = BayesNet()
# bn.load_from_bifxml('examples/UseCase.BIFXML')
# bn.draw_structure()
# var = bn.get_all_variables()
#
# cpts = bn.get_all_cpts()


def cpt_latex(fname: str):
    bn = BayesNet()
    bn.load_from_bifxml(USE_CASE_FILE)
    var = bn.get_all_variables()

    content = ""
    for element in var:
        cpt = bn.get_cpt(element)
        content += cpt.to_latex(index=False) + "\n"

    with open(fname, "w") as f:
        f.write(content)
    print(f"wrote file: '{fname}'")


# print(type(cpts))


def main():
    fname = os.path.join(SCRIPT_DIR, "tables.tex")
    cpt_latex(fname)

    # br = BNReasoner(USE_CASE_FILE)
    # visualize(br, node_size=500)
    # interesting_queries()
    # rand_network()


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
