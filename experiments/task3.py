#!/usr/bin/env python3
from typing import Union, MutableSet, Dict, List, Any, Tuple
import networkx as nx
import pandas as pd
from copy import deepcopy
from enum import Enum
import os
from pgmpy.readwrite import XMLBIFReader

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
    vars = bn.get_all_variables()

    content = ""
    for var in vars:
        content += f"% {var}\n\n"
        content += f"CPT for Node '{var}'\n"
        cpt = bn.get_cpt(var)
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

    interesting_queries(Ordering.MIN_FILL)


# TODO: populate this function with all the interesting queries we care about
def interesting_queries(ordering_method: Ordering):
    # all_vars = [
    #    "bad-weather",
    #    "being-drunk",
    #    "car-accident",
    #    "decreased-alertness",
    #    "distractions",
    #    "on-time",
    #    "risky-driver",
    #    "under-25",
    #    "use-car",
    #    "woman",
    # ]

    br = BNReasoner(USE_CASE_FILE)
    all_vars = sorted(br.bn.get_all_variables())

    print(f"running some queries on: '{USE_CASE_FILE}'")
    print(f"NOTE: all variables:\n{all_vars}\n")

    # note: if you want to be really careful, create a deep copy of br before each query
    # but the querying methods shouldn't modify br)
    # visualize(br, node_size=500)

    # https://pgmpy.org/examples/Inference%20in%20Discrete%20Bayesian%20Networks.html#Inference-in-Discrete-Bayesian-Network
    model = XMLBIFReader(USE_CASE_FILE).get_model()
    from pgmpy.inference import VariableElimination

    model_infer = VariableElimination(model)

    res = br.marginal_distribution(
        {"woman"},
        pd.Series({}),
        ordering_method=ordering_method,
    )
    print("\nmarginal_dist query0:\n")
    print(res)

    # compare to result from pgmpy:
    fact = model_infer.query(variables=["woman"], evidence={})
    import pdb

    pdb.set_trace()

    print("pgympy result:")
    print(fact.variables)
    print(fact.values)

    res = br.marginal_distribution(
        {"arriving-on-time", "woman"},
        pd.Series({"bad-weather": True}),
        ordering_method=ordering_method,
    )
    print("\nmarginal_dist query1:\n")
    print(res)

    res = br.marginal_distribution(
        {"on-time"},
        pd.Series({"bad-weather": True}),
        ordering_method=ordering_method,
    )
    print("\nmarginal_dist query2:\n")
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
        pd.Series({"car-accident": True}),
        ordering_method=ordering_method,
    )
    print("\nMPE query:\n")
    print(res)
    import pdb

    pdb.set_trace()

    # print('MAP')


if __name__ == "__main__":
    main()
