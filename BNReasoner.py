from typing import Union, MutableSet, Dict, List, Any, Tuple
from BayesNet import BayesNet
import networkx as nx
import pandas as pd
from copy import deepcopy
from enum import Enum

# Evidence = Dict[str, Any]
Evidence = pd.Series  # e.g.: pd.Series({"A": True, "B": False})


class Ordering(Enum):
    """Defines types of ordering heuristics/methods."""

    MIN_FILL = "min_fill"
    MIN_DEG = "min_degree"


class BNReasoner:
    bn: BayesNet

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
            assert type(net) == BayesNet
            self.bn = net

    def deepcopy(self) -> "BNReasoner":
        """Clones this object, returning a deep copy"""
        return BNReasoner(deepcopy(self.bn))

    def _edge_pruning(self, Z: MutableSet[str]) -> None:
        """
        Delete all outgoing edges in network from nodes Z.
        (Note: doesn't update CPTs).
        """
        for node in Z:
            children = self.bn.get_children(node)
            for child in children:
                self.bn.del_edge([node, child])

    def _node_pruning(self, r_nodes: MutableSet[str]) -> None:
        """
        Delete leaf nodes in network which are not in relevant nodes.
        (Note: doesn't update CPTs).
        """
        nodes = self.bn.get_all_variables()

        # (this probably doesn't need to be a loop)
        while True:
            leaf_nodes = [
                node
                for node in nodes
                if self.bn.get_children(node) == []
                if node not in r_nodes
            ]

            if leaf_nodes == []:
                break

            for node in leaf_nodes:
                self.bn.del_var(node)
                nodes.remove(node)

    def d_separated(self, X: MutableSet[str], Y: MutableSet[str], Z: MutableSet[str]):
        """
        Given three sets of variables X, Y, and Z, determine whether X is d-separated
        of Y given Z by pruning network iteratively: Delete all outgoing edges of nodes
        in Z. Delete every leaf node W which is not in sets of nodes X or Y or Z.
        """

        br = self.deepcopy()  # create deep copy of self we can destructively edit

        br._edge_pruning(Z)
        br._node_pruning(set.union(X, Y, Z))

        # create list of paths
        paths = [(x, y_test) for x in X for y_test in Y]

        for x, y in paths:

            # check whether x and y are not d-separated (i.e. there is a path)
            if nx.has_path(nx.to_undirected(br.bn.structure), x, y):
                return False

        return True

    def independent(self, X: MutableSet[str], Y: MutableSet[str], Z: MutableSet[str]):
        """
        Given three sets of variables X, Y, and Z, determine whether X is
        independent of Y given Z.
        """
        return self.d_separated(X, Y, Z)

    """
    Possibly TODO: write factor reduction in separate function to use again for
    posterior marginals
    """

    def network_pruning(self, Q: MutableSet[str], e: Evidence) -> None:
        """
        Given a set of query variables Q and evidence e, simplify the network
        structure, so that queries of the form P(Q|E) can still be correctly
        calculated.
        """
        # br = self.deepcopy()  # create deep copy of self we can destructively edit

        E = set(e.keys())

        # perform factor reduction on all child cpts
        for var, value in e.items():
            children = self.bn.get_children(var)

            for child_var in children:
                cpt = self.bn.get_cpt(child_var)
                # zero out rows where var != E[var]:
                new_cpt = BayesNet.reduce_factor(e, cpt)
                # sum out var:
                new_cpt = BNReasoner.marginalize(new_cpt, var)
                self.bn.update_cpt(child_var, new_cpt)

            # also filter down rows of var (and reset index to play nice with tests)
            cpt = self.bn.get_cpt(var)
            cpt = (cpt[cpt[var] == value]).reset_index(drop=True)
            self.bn.update_cpt(var, cpt)

        """TODO: edge_pruning needs (?) to come before reduce factor"""
        # remove outgoing edges from vars in E
        self._edge_pruning(E)
        # remove any leaf nodes not appearing in Q or e
        self._node_pruning(set.union(Q, E))

    @staticmethod
    def marginalize(
        f: pd.DataFrame,
        X: str,
    ):
        """
        Given a factor and a variable X, compute the CPT in which X is summed-out.

        :param f: the factor to maginalize.
        :param X: name of the variable to sum out.
        """

        # e.g. ['B', 'C', 'D']
        all_vars = [v for v in f.columns.values.tolist() if v != "p"]
        # e.g. ['B', 'C']
        new_vars = [v for v in all_vars if v not in set([X, "p"])]

        cpt = pd.DataFrame([], columns=new_vars + ["p"])
        data = {}
        for idx, row in f.iterrows():
            key = row[new_vars].to_string()  # string representing vals in rows
            # store index of row with these values that has max p value
            if key not in data:
                data[key] = []
            data[key].append(idx)

        idx_lists = list(data.values())  # list of row indices to keep
        for indices in idx_lists:
            # indices is a list of row indices in f where new_vars are identical
            p = sum(f.loc[indices, "p"])
            vals = f.loc[indices[0], new_vars].tolist()
            # add new row to final cpt
            # cpt = cpt.append(vals + [p], columns=new_vars + ["p"])
            cpt = cpt.append(
                pd.Series(vals + [p], index=cpt.columns), ignore_index=True
            )

        cpt = cpt.reset_index(drop=True)
        return deepcopy(cpt)  # just in case

    @staticmethod
    def max_out(f: pd.DataFrame, X: str):
        """
        Given a factor and a variable X, compute the CPT in which X is maxed-out.
        Keep track of which instantiation of X led to the maximized value.

        :param f: the factor to max-out.
        :param X: name of the variable to max-out.
        """
        # e.g. ['B', 'C', 'D']
        all_vars = [v for v in f.columns.values.tolist() if v != "p"]
        # e.g. ['B', 'C']
        new_vars = [v for v in all_vars if v not in set([X, "p"])]

        # cpt = pd.DataFrame([], columns=new_vars + ["p"])
        data = {}
        for idx, row in f.iterrows():
            key = row[new_vars].to_string()  # string representing vals in rows
            # store index of row with these values that has max p value
            if key not in data:
                data[key] = idx
            else:
                if f.loc[data[key], "p"] < row["p"]:
                    data[key] = idx  # found row (of same vals) with larger p

        keep_idxs = list(data.values())  # list of row indices to keep
        cpt = deepcopy(f.loc[keep_idxs, :])
        del cpt[X]
        cpt.reindex()
        cpt = cpt.reset_index(drop=True)
        return cpt

    @staticmethod
    def multiply_factors(f: pd.DataFrame, g: pd.DataFrame):
        """
        Given two factors f and g, compute the multiplied factor h = f * g.

        :param f, g: cpts of factors f and g
        """
        merge_on = list(f.columns & g.columns)
        merge_on.remove("p")

        # https://stackoverflow.com/questions/54657907/pandas-merging-two-dataframes-on-multiple-columns-and-multiplying-result
        h = f.merge(g, on=merge_on, how="outer")
        h["p"] = h["p_x"] * h["p_y"]

        return h.drop(["p_x", "p_y"], axis=1)

    def get_ordering(self, vars: MutableSet[str], method: Ordering) -> List[str]:
        """
        Suggests the order by which to sum out vars (according to the desired heuristic/method).
        Note that when computing the ordering, in the case of a tie (e.g. two nodes have
        the same degree) alphabetical order is used as the tie-breaker.
        # TODO: consider merging get_ordering into "variable elimination" function?
        """
        net = deepcopy(self.bn)
        to_remove = set(vars)
        assert to_remove.issubset(set(net.get_all_variables()))

        def get_new_interactions(
            ig: nx.Graph, var: str, add_edges=False
        ) -> List[Tuple[str, str]]:
            """
            Helper function returning a dict of new interactions if var were removed
            from the provided interaction graph.
            Optionally also adds new interactions if add_edges == True.
            """
            newi = dict()
            neighbors = list(ig.neighbors(var))

            for n1 in neighbors:
                for n2 in neighbors:
                    key = ",".join(sorted([n1, n2]))  # e.g. "A,B"
                    if n1 == n2 or ig.has_edge(n1, n2) or key in newi:
                        continue
                    newi[key] = (n1, n2)
                    if add_edges:
                        ig.add_edge(n1, n2)
            return list(newi.values())

        ordering = []
        if method == Ordering.MIN_DEG:
            ig = net.get_interaction_graph()
            # simulate summing out (on ig) until all desired variables are removed
            while len(to_remove) > 0:
                remaining = sorted([n for n in ig.nodes if n in vars])
                # determine which var to remove next:
                var = min(remaining, key=lambda v: ig.degree[v])

                # add an edge between every neighbor of var that's not already connected
                neighbors = list(ig.neighbors(var))
                for n1 in neighbors:
                    for n2 in neighbors:
                        if n1 == n2 or ig.has_edge(n1, n2):
                            continue
                        ig.add_edge(n1, n2)
                        # note that if we were actually summing out var, we'd do factor multiplication here to update the cpts...

                ig.remove_node(var)
                ordering.append(var)
                to_remove.discard(var)

        elif method == Ordering.MIN_FILL:
            ig = net.get_interaction_graph()
            while len(to_remove) > 0:
                remaining = sorted([n for n in ig.nodes if n in vars])
                # inters = {v: get_new_interactions(ig, v) for v in remaining}
                # print(f"\ninters = {inters}")
                var = min(remaining, key=lambda v: len(get_new_interactions(ig, v)))

                # apply changes
                get_new_interactions(ig, var, add_edges=True)
                ig.remove_node(var)
                # print(f"removing node {var}")

                ordering.append(var)
                to_remove.discard(var)
        else:
            raise ValueError(f"ordering method not implemented '{method}'")

        return ordering

    def variable_elimination(self, vars: MutableSet[str], method=Ordering.MIN_DEG):
        """
        Sum out a given set of variables by using variable elimination.
        """
        ordered = self.get_ordering(vars, method)

        for var in ordered:
            root_cpt = self.bn.get_cpt(var)

            cpts = [self.bn.get_cpt(child) for child in self.bn.get_children(var)]

            for cpt in cpts:
                new_cpt = BNReasoner.multiply_factors(root_cpt, cpt)
                new_cpt = BNReasoner.marginalize(new_cpt, var)

        return new_cpt

    def MPE(
        self, e: Evidence, ordering_method=Ordering.MIN_DEG
    ) -> Tuple[Evidence, float]:
        """
        Given evidence e, compute the most probable explanation.
        :param e: a series of assignments as tuples. E.g.: pd.Series({"A": True, "B": False})
        :param ordering_method: (optional) enum indicating which ordering method to use.

        :return a tuple containing a dictionary of assignments (for all variables in the network), and a probabilitity.
        """
        br = self.deepcopy()  # create deep copy of self we can destructively edit

        all_vars = set(self.bn.get_all_variables())

        known_vars = set(list(e.keys()))
        Q = all_vars - known_vars  # list of unknown vars

        br.network_pruning(Q, e)

        # br = BNReasoner._edge_pruning(br, known_vars)
        # no node_pruning for MEP because Q + e = all_vars
        # net = BNReasoner.node_pruning(net, unknown_vars)

        # compute an ordering for variable removal
        ordering = br.get_ordering(Q, method=ordering_method)  # Q or something else??

        # TODO do removals, computing new cpts (use extended factors somehow)...
        # also use BayesNet.reduce_factor and get_compatible_instantiations_table

    def MAP(
        self, Q: MutableSet[str], e: Evidence, ordering_method=Ordering.MIN_DEG
    ) -> Tuple[Evidence, float]:
        """
        Compute the maximum a-posteriory instantiation + value of query variables Q given (possibly empty) evidence e.

        :param Q: a set of variable names being queried.
        :param e: a series of assignments as tuples. E.g.: pd.Series({"A": True, "B": False})
        :param ordering_method: (optional) enum indicating which ordering method to use.
        """
        pass
