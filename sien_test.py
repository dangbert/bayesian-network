from typing import Union, MutableSet, Dict, List, Any, Tuple
from BayesNet import BayesNet
from BNReasoner import BNReasoner
import networkx as nx
import pandas as pd
from copy import deepcopy
from enum import Enum


# bn = BayesNet()
# bn.load_from_bifxml('examples/UseCase.BIFXML')
# bn.draw_structure()
# var = bn.get_all_variables()
#
# cpts = bn.get_all_cpts()

def cpt_latex():
    bn = BayesNet()
    bn.load_from_bifxml('examples/UseCase.BIFXML')
    var = bn.get_all_variables()

    for element in var:
        cpt = bn.get_cpt(element)
        print(cpt.to_latex(index = False))

cpt_latex()
# print(type(cpts))
