from typing import Union, MutableSet, Dict, List
from BayesNet import BayesNet
import networkx as nx
import pandas as pd
from copy import deepcopy



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

    @staticmethod
    def edge_pruning(network: BayesNet, Z: MutableSet[str]):
        """
        Delete all outgoing edges from Z.
        """
        for node in Z:
            children = network.get_children(node)
            for child in children:
                network.del_edge([node, child])

    @staticmethod
    def node_pruning(network: BayesNet, nodes: List[str], r_nodes: MutableSet[str]):
        """
        Delete leaf nodes which are not in relevant nodes.
        """
        while True:
            leaf_nodes = [node for node in nodes if network.get_children(node) == [] if node not in r_nodes]

            if leaf_nodes == []:
                break

            for node in leaf_nodes:
                network.del_var(node)
                nodes.remove(node)


    def d_separated(self, X: MutableSet[str], Y: MutableSet[str], Z: MutableSet[str]):
        """
        Prune network iteratively. Deletes all outgoing edges of nodes in Z. Deletes
        every leaf node W which is not in sets of nodes X or Y or Z.
        """
        bn_copy = deepcopy(self.bn)

        relevant_nodes = set.union(X, Y, Z)
        nodes = bn_copy.get_all_variables()

        BNReasoner.edge_pruning(bn_copy, Z)
        BNReasoner.node_pruning(bn_copy, nodes, relevant_nodes)

        # create list of paths
        paths = [(x, y_test) for x in X for y_test in Y]

        for x, y in paths:

            # check whether x and y are not d-separated (i.e. there is a path)
            if nx.has_path(nx.to_undirected(bn_copy.structure), x, y):
                return False

        return True

    def independent(self, X: MutableSet[str], Y: MutableSet[str], Z: MutableSet[str]):
        """
        Determine whether X is independent of Y given Z.
        """
        return self.d_separated(X, Y, Z)

    def network_pruning(self, Q: MutableSet[str], e: Dict[str, str]):
        """
        Given a set of query variables Q and evidence e, simplify the network
        structure.
        """
        bn_copy = deepcopy(self.bn)

        # relevant_nodes = set.union(X, Y, Z)
        # nodes = bn_copy.get_all_variables()


        BNReasoner.node_pruning(bn_copy, nodes, relevant_nodes)

        E = set(e.keys())
        BNReasoner.edge_pruning(bn_copy, E)

        # TO DO: implement factor reduction

        # node_pruning(bn_copy, nodes, relevant_nodes)
        for key, value in e.items():
            cpt = bn_copy.get_cpt(key)
            cpt = cpt[cpt[key]] != value
            print(cpt)

            # Siens doen vanaf hier so check it:
            # update the cpts in the BN
            bn_copy.update_cpt(key, cpt)

        # ehh? kijk hier even naar want r_nodes hier is nu niet meer XYZ zoals we
        # wel willen in de node_pruning function
        BNReasoner.node_pruning(bn_copy, nodes, r_nodes)



        pass

    @staticmethod
    def marginalize(
        f: pd.DataFrame,
        X: str,
    ):
        """
        Given a factor and a variable X, compute the CPT in which X is summed-out.
        TODO: may not work for higher dimensional factors

        :param f: the factor to maginalize.
        :param X: name of the variable to sum out.
        """
        # indices = [idx for _, idx in enumerate(f[X])]

        cpt = pd.DataFrame()
        for var in f.keys():
            if var in set(["p", X]):
                continue
            newP = []
            cpt[var] = pd.Series(
                list(set(f[var]))
            )  # all vals var takes on... e.g. {False, True}

            # TODO: swap to list comp
            # for val in set(f[var]):
            for val in cpt[var]:  # all vals var takes on... e.g. {False, True}
                # cpt[var].append(sum(f.loc[f[var] == val]["p"]))
                newP.append(sum(f.loc[f[var] == val]["p"]))

            cpt["p"] = pd.Series(newP)
        return cpt
