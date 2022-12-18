#!/usr/bin/env python3
from typing import Union, MutableSet, Dict, List, Any, Tuple
import networkx as nx
import pandas as pd
from copy import deepcopy
from enum import Enum
import os
from pgmpy.readwrite import XMLBIFReader
import logging
import warnings

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
        content += f"CPT for Node '{var}':\n\n"
        cpt = bn.get_cpt(var)
        content += cpt.to_latex(index=False) + "\n"

    with open(fname, "w") as f:
        f.write(content)
    print(f"wrote file: '{fname}'")


# print(type(cpts))


def main():
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    log_level = logging.DEBUG
    logging.basicConfig(format=FORMAT, level=log_level)

    # hide annyoying pandas deprecation warnings
    warnings.filterwarnings("ignore")

    fname = os.path.join(SCRIPT_DIR, "_tables.tex")
    cpt_latex(fname)

    # br = BNReasoner(USE_CASE_FILE)
    # visualize(br, node_size=500)

    fname = os.path.join(SCRIPT_DIR, "_part3.tex")
    interesting_queries(Ordering.MIN_FILL, fname)


# TODO: populate this function with all the interesting queries we care about
def interesting_queries(ordering_method: Ordering, fname: str):
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

    float_format = "{:0.4f}".format
    NEWLINES = "\n\\vfill\\vfill\n"
    latex = NEWLINES

    def process_result(name: str, res, header: bool = True):
        nonlocal latex
        nonlocal float_format

        print(f"\n{name}")
        if type(res) == tuple:
            prob, res = res
            print(f"prob = {prob:0.4f}")
            print(res)
            latex += f"{name}\n\nprob = {prob:0.4f}\n\\vfill\n{res.to_latex(index=True, float_format=float_format, header=header)}{NEWLINES}{NEWLINES}"
        else:
            print(res)
            latex += f"{name}\n\\vfill\n{res.to_latex(index=False, float_format=float_format, header=header)}{NEWLINES}{NEWLINES}"

    model_infer = VariableElimination(model)

    res = br.deepcopy().marginal_distribution(
        {"car-accident"},
        pd.Series({}),
        ordering_method=ordering_method,
    )
    print("\nprior probability of 'car-accident':")
    print(res)

    latex += res.to_latex(index=False) + "\n"

    res = br.deepcopy().marginal_distribution(
        {"car-accident"},
        pd.Series({"under-25": True, "woman": True}),
        ordering_method=ordering_method,
    )
    name = (
        "posterior probability of being in a car accident given under-25 and a woman:"
    )
    process_result(name, res)

    res = br.deepcopy().MAP(
        {"bad-weather"},
        pd.Series({"car-accident": True}),
        ordering_method=ordering_method,
    )
    name = "\nmost likely state of the node 'bad-weather' given someone is in a car accident:"
    process_result(name, res, header=False)

    res = br.deepcopy().MPE(
        pd.Series({"car-accident": True}),
        ordering_method=ordering_method,
    )
    name = "MPE of a car-accident:"
    process_result(name, res, header=False)

    # res = br.MAP(set(all_vars), pd.Series({}))
    # fact = model_infer.query(variables=all_vars, evidence={})
    # print("MAP")

    # res = br.deepcopy().marginal_distribution(
    #    {"woman"},
    #    pd.Series({}),
    #    ordering_method=ordering_method,
    # )
    # print("\prior probability of 'woman':\n")
    # print(res)

    # compare to result from pgmpy:
    # fact = model_infer.query(variables=["woman"], evidence={})
    # print("pgympy result:")
    # print(fact.variables)
    # print(fact.values)

    # res = br.deepcopy().marginal_distribution(
    #    {"on-time", "woman"},
    #    pd.Series({"bad-weather": True}),
    #    ordering_method=ordering_method,
    # )
    # print("\nmarginal_dist query1:\n")
    # print(res)

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

    with open(fname, "w") as f:
        f.write(latex)
    print(f"\nwrote file: '{fname}'")


if __name__ == "__main__":
    main()
