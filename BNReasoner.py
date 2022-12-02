from typing import Union, MutableSet
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

    def pruning(self, nodes: Tuple(MutableSet[str], MutableSet[str], MutableSet[str])):
        """
        Prune network iteratively. Deletes all outgoing edges of nodes in Z. Deletes
        every leaf node W which is not in sets of nodes X or Y or Z.
        """
        X, Y, Z = node_set[0], node_set[1], node_set[2]
        relevant_nodes = set.union(X, Y, Z)

        nodes = self.bn.get_all_variables()

        # edge pruning
        for node in Z:
            children = self.bn.get_children(node)
            for child in children:
                self.bn.del_edges([node, child])

        # node pruning
        while True:
            leaf_nodes = [node for node in nodes if self.bn.children(node) == []]

            if leaf_nodes = []:
                break

            for node in leaf_nodes:
                if node not in relevant_nodes:
                    self.bn.del_var(node)


    def path_type(self, x, y):
        """
        Determines whether a specific node n is a sequence, fork, or collider
        on the path from x to y.
        """
        # confirm there is a path between x and y
        # if nx.is_path(self.bn.structure, [x, y]):

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
