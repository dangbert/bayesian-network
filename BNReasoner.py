from typing import Union
from BayesNet import BayesNet


class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        else:
            self.bn = net

    # TODO: This is where your methods should go
    def path_type(self, x, y):
        """
        Determines whether a specific node z is a sequence, fork, or collider
        on the path from x to y.
        """
        pass

    def d_blocked(self, x, y):
        """
        Checks whether the path between nodes x and y are d-blocked by z.
        """
        pass

    def d_seperated(self, x, y):
        """
        Checks whether all paths between nodes x and y are d-blocked.
        """
