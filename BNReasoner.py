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

    def d_separated(self, X: MutableSet[str], Y: MutableSet[str], Z: MutableSet[str]):
        """
        Prune network iteratively. Deletes all outgoing edges of nodes in Z. Deletes
        every leaf node W which is not in sets of nodes X or Y or Z.
        """
        relevant_nodes = set.union(X, Y, Z)

        nodes = self.bn.get_all_variables()

        # edge pruning i.e. delete all outgoing edges (from Z)
        for node in Z:
            children = self.bn.get_children(node)
            for child in children:
                self.bn.del_edges([node, child])

        # node pruning i.e. delete leaf nodes
        while True:
            leaf_nodes = [node for node in nodes if self.bn.children(node) == []]

            if leaf_nodes = []:
                break

            for node in leaf_nodes:
                if node not in relevant_nodes:
                    self.bn.del_var(node)

        # create list of paths
        paths = [(x, y) for x in X and y in Y]

        for x, y in paths:

            # check whether x and y are not d-separated (i.e. there is a path)
            if nx.has_path(nx.to_undirected(self.bn.structure), x_node, y_node):
                return False

        return True

    def independence(self, X: MutableSet[str], Y: MutableSet[str], Z: MutableSet[str]):
        """
        Determine whether X is independent of Y given Z.
        """
        if d_separated(X, Y, Z):
            return True


    def network_pruning(self, Q: , e: ):
        """
        Given a set of query variables Q and evidence e, simplify the network
        structure.
        """
        pass
