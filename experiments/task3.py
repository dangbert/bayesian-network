#!/usr/bin/env python3
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

    interesting_queries(Ordering.MIN_FILL)

    # br = BNReasoner(USE_CASE_FILE)
    # visualize(br, node_size=500)


# TODO: populate this function with all the interesting queries we care about
def interesting_queries(ordering_method: Ordering):

    br = BNReasoner(USE_CASE_FILE)
    all_vars = sorted(br.bn.get_all_variables())

    print(f"running some queries on: '{USE_CASE_FILE}'")
    print(f"NOTE: all variables:\n{all_vars}\n")

    # note: if you want to be really careful, create a deep copy of br before each query
    # but the querying methods shouldn't modify br)
    # visualize(br, node_size=500)

    # TODO: debug why this one fails cause len(Q) == 1
    # res = br.marginal_distribution(
    #    {"arriving-on-time", "woman"},
    #    pd.Series({"bad-weather": True}),
    #    ordering_method=ordering_method
    # )

    res = br.marginal_distribution(
        {"on-time"},
        pd.Series({"bad-weather": True}),
        ordering_method=ordering_method,
    )
    print("\nmarginal_dist query:\n")
    print(res)

    # TODO: debug why this one also fails
    # res = br.MAP(
    #    {"woman", "being-drunk"},
    #    pd.Series({"under-25": True}),
    #    ordering_method=ordering_method,
    # )
    # print("\nMAP query:\n")
    # print(res)

    # res = br.MAP(
    #    {"being-drunk", "distractions"},
    #    pd.Series({"woman": True}),
    #    ordering_method=ordering_method,
    # )
    # print("\nMAP query2:\n")
    # print(res)

    res = br.MPE(
        pd.Series({"under-25": True, "risky-driver": True}),
        ordering_method=ordering_method,
    )
    print("\nMPE query:\n")
    print(res)

    # print('MAP')


if __name__ == "__main__":
    main()
