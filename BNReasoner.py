from typing import Union, MutableSet, Dict, List, Any, Tuple
from BayesNet import BayesNet
import networkx as nx
import pandas as pd
from copy import deepcopy
from enum import Enum

Evidence = pd.Series  # e.g.: pd.Series({"A": True, "B": False})

# prefix used to denote columns representing a known instantiation
INS = "_ins_"


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

    def get_predecessors(self, var: str) -> List[str]:
        """
        Returns the predecessors of the variable in the graph.

        :param var: Variable to get the predecessors from.
        :return: List of predecessors.
        """
        return [p for p in nx.predecessors(var)]

    def _non_queried_variables(self, Q: MutableSet[str]) -> List[str]:
        """
        Returns a list of all variables in the graph excluding queried variables.
        """
        return set(self.bn.get_all_variables()) - Q

    def _edge_pruning(self, Z: MutableSet[str]) -> None:
        """
        Delete all outgoing edges in network from nodes Z.
        (Note: doesn't update CPTs).

        :param Z: Set of nodes whose outgoing edges will be deleted.
        """
        for node in Z:
            children = self.bn.get_children(node)
            for child in children:
                self.bn.del_edge([node, child])

    def _node_pruning(self, r_nodes: MutableSet[str]) -> None:
        """
        Delete leaf nodes in network which are not in relevant nodes.
        (Note: doesn't update CPTs).

        :param r_nodes: Set of nodes not to be deleted even if leaf node.
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

    def _apply_evidence(self, e: Evidence, condition: bool = False) -> None:
        """
        Update CPTs (in place) based on evidence.
        TODO: rename condition variable to something better!!!
        """
        if e.empty:
            return

        # perform factor reduction on all child cpts
        for var, value in e.items():
            children = self.bn.get_children(var)

            for child_var in children:
                cpt = self.bn.get_cpt(child_var)
                # zero out rows where var != E[var]:
                new_cpt = BayesNet.reduce_factor(e, cpt)
                # sum out var:
                if condition:
                    new_cpt = BayesNet.get_compatible_instantiations_table(e, new_cpt)
                else:
                    new_cpt = BNReasoner.marginalize(new_cpt, var)

                self.bn.update_cpt(child_var, new_cpt)

            # also filter down rows of var (and reset index to play nice with tests)
            cpt = self.bn.get_cpt(var)
            cpt = (cpt[cpt[var] == value]).reset_index(drop=True)
            # TODO: set prob to 1.0 for every remaining row in cpt?
            self.bn.update_cpt(var, cpt)

    def network_pruning(self, Q: MutableSet[str], e: Evidence) -> None:
        """
        Given a set of query variables Q and evidence e, simplify the network
        structure, so that queries of the form P(Q|E) can still be correctly
        calculated.
        """
        # br = self.deepcopy()  # create deep copy of self we can destructively edit
        E = set(e.keys())

        # perform factor reduction
        self._apply_evidence(e)

        # remove outgoing edges from vars in E
        self._edge_pruning(E)

        # remove any leaf nodes not appearing in Q or e
        self._node_pruning(set.union(Q, E))

    @staticmethod
    def marginalize(f: pd.DataFrame, X: str) -> pd.DataFrame:
        """
        Given a factor and a variable X, compute the CPT in which X is summed-out.

        :param f: the factor to maginalize.
        :param X: name of the variable to sum out.
        """
        vars = [v for v in f.columns if v not in set([X, "p"])]
        cpt = f.groupby(vars).sum().reset_index()
        return cpt.drop(X, axis=1)

    @staticmethod
    def max_out(f: pd.DataFrame, X: str) -> pd.DataFrame:
        """
        Given a factor and a variable X, compute the CPT in which X is maxed-out.
        Keeps track of which instantiation of X led to the maximized value (by storing it in a column named f"{INS}{X}").

        :param f: the factor to max-out.
        :param X: name of the variable to max-out.
        """
        vars = [
            v for v in f.columns if v not in set([X, "p"]) and not v.startswith(INS)
        ]

        # cpt = f.groupby(vars)["p"].max().reset_index()
        max_index = f.groupby(vars)["p"].idxmax()
        cpt = f.loc[max_index]
        cpt = cpt.reset_index(drop=True)

        # keep track of instantiations
        extended = cpt[X]
        # remove X column from dataframe
        cpt = cpt.drop(X, axis=1)
        # add column to end denoting instations of X
        cpt[f"{INS}{X}"] = extended
        return cpt

    @staticmethod
    def multiply_factors(f: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
        """
        Given two factors f and g, compute the multiplied factor h = f * g.

        :param f, g: cpts of factors f and g
        :returns: cpt of multiplied factor h
        """
        merge_on = list(f.columns & g.columns)
        merge_on.remove("p")

        if len(merge_on) > 0:
            h = f.merge(g, on=merge_on, how="outer")
        else:
            h = f.merge(g, how="cross")
        h["p"] = h["p_x"] * h["p_y"]
        return h.drop(["p_x", "p_y"], axis=1)

    def get_ordering(self, vars: MutableSet[str], method: Ordering) -> List[str]:
        """
        Suggests the order by which to sum out vars (according to the desired heuristic/method).
        Note that when computing the ordering, in the case of a tie (e.g. two nodes have
        the same degree) alphabetical order is used as the tie-breaker.
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

    def variable_elimination(
        self, Q: MutableSet[str], method=Ordering.MIN_DEG
    ) -> pd.DataFrame:
        """
        Sum out a given set of variables (all_vars - Q) by using variable elimination. Calculate
        and return prior marginal.
        TODO: should be consistent about naming "method" vs "ordering_method" working.
        """
        ordered = self.get_ordering(self._non_queried_variables(Q), method)

        all_cpts = list(self.bn.get_all_cpts().values())

        res = None  # running scratchpad of result
        for var in ordered:
            # find all cpts containing var
            rel_cpts = [cpt for cpt in all_cpts if var in cpt.columns]

            for cpt in rel_cpts:
                if res is None:
                    res = cpt
                    continue
                res = BNReasoner.multiply_factors(res, cpt)

            # if len(res.columns) - 1 == len(Q):
            #    import pdb

            #    pdb.set_trace()
            #    return res
            # sum out
            res = BNReasoner.marginalize(res, var)
            # print(f"after summed out\n{res}")

            # update list of remaining cpts
            all_cpts = [cpt for cpt in all_cpts if var not in cpt.columns]

        # print(f"final_cpt:\n {cpt}")
        return res
        # TODO: consider actually returning this:
        return [res] + all_cpts

    def marginal_distribution(
        self, Q: MutableSet[str], e: Evidence, ordering_method=Ordering.MIN_DEG
    ) -> pd.DataFrame:
        """
        Given query variables Q and possibly empty evidence e, compute the marginal
        distribution P(Q|e). Note that Q is a subset of the variables in the Bayesian
        network X with Q ⊂ X but can also be Q = X.
        Note when evidence is empty, we are getting the "prior marginal probability"

        :returns: dataframe containing P(not Q|e) and P(Q|e)
        """
        br = self.deepcopy()  # create deep copy of self we can destructively edit
        # reduce all factors w.r.t. e
        br._apply_evidence(e, condition=True)

        # TODO: what if  Q = X? (add test for this)

        # compute joint marginal PR (Q & e) via variable elim
        joint_marginal = br.variable_elimination(Q, ordering_method)

        if not e.empty:

            # sum out C to get probability of e
            prob_e = joint_marginal["p"].sum()
            # "normalize" to obtain Pr(Q, e) (see "Posterior Marginal" slides)
            #   cause for Bayes theorem you have to divide by p(e)
            joint_marginal["p"] = joint_marginal["p"].div(prob_e)

        return joint_marginal

    def MPE(
        self, e: Evidence, ordering_method=Ordering.MIN_DEG
    ) -> Tuple[float, Evidence]:
        """
        Given evidence e, compute the most probable explanation.
        :param e: a series of assignments as tuples. E.g.: pd.Series({"A": True, "B": False})
        :param ordering_method: (optional) enum indicating which ordering method to use.

        :return a tuple containing a dictionary of assignments (for all variables in the network), and a probabilitity.
        """
        br = self.deepcopy()  # create deep copy of self we can destructively edit

        all_vars = set(br.bn.get_all_variables())
        known_vars = set(list(e.keys()))
        Q = all_vars - known_vars  # list of unknown vars
        br.network_pruning(Q, e)  # (also applies evidence)

        # compute an ordering for variable removal
        ordering = br.get_ordering(Q, ordering_method)
        all_cpts = list(br.bn.get_all_cpts().values())

        # now we do variable_elimination but by maxing_out
        res = None
        for var in ordering:
            # find all cpts containing var
            rel_cpts = [cpt for cpt in all_cpts if var in cpt.columns]

            # res = None
            for cpt in rel_cpts:
                if res is None:
                    res = cpt
                    continue
                res = BNReasoner.multiply_factors(res, cpt)

            # max out var
            res = BNReasoner.max_out(res, var)

            # update list of remaining cpts
            all_cpts = [cpt for cpt in all_cpts if var not in cpt.columns]

        # final pass through remaining CPTs
        for cpt in all_cpts:
            res = BNReasoner.multiply_factors(res, cpt)
        for i, var in enumerate(e.keys()):
            if i != len(e.keys()) - 1:
                res = BNReasoner.max_out(res, var)
            else:
                # can't max_out last var cause its the only one left!
                max_idx = res["p"].idxmax()
                res = res.iloc[max_idx : (max_idx + 1)]
                res = res.rename(columns={var: f"{INS}{var}"})

        assert len(res) == 1, f"expect 1 row left, found {len(res)}"
        p = res["p"][0]
        ins = {}
        for c in list(res.columns):
            if c.startswith(INS):
                ins[c[len(INS) :]] = res[c][0]
        ins = pd.Series(ins).sort_index()
        return p, ins

    def MAP(
        self, Q: MutableSet[str], e: Evidence, ordering_method=Ordering.MIN_DEG
    ) -> Tuple[float, Evidence]:
        """
        Compute the maximum a-posteriori instantiation + value of query variables Q given (possibly empty) evidence e.

        :param Q: a set of variable names being queried.
        :param e: a series of assignments as tuples. E.g.: pd.Series({"A": True, "B": False})
        :param ordering_method: (optional) enum indicating which ordering method to use.
        """
        br = self.deepcopy()  # create deep copy of self we can destructively edit

        # step 1: reduce w.r.t. e
        br._apply_evidence(e, condition=True)

        # step 2: repeatedly multiply and sum out
        res = br.variable_elimination(Q, ordering_method)

        # all_cpts = list(br.bn.get_all_cpts().values())

        # step 3: max out Q
        for var in Q:
            cpt = br.bn.get_cpt(var)
            res = self.multiply_factors(res, cpt)

            if len([c for c in res.columns if not c.startswith(INS)]) > 2:
                res = br.max_out(res, var)
            else:
                # can't max_out last var cause its the only one left!
                max_idx = res["p"].idxmax()
                res = res.iloc[max_idx : (max_idx + 1)]
                res = res.rename(columns={var: f"{INS}{var}"})

        assert len(res) == 1, f"expect 1 row left, found {len(res)}"
        p = res["p"][0]
        ins = {}
        for c in list(res.columns):
            if c.startswith(INS):
                ins[c[len(INS) :]] = res[c][0]
        ins = pd.Series(ins).sort_index()
        return p, ins

    # @staticmethod
    # def separate_instantations(f: pd.DataFrame) -> Tuple[pd.DataFrame, Evidence]:
    #    """Given a dataframe of one row that contains instation columns, split it into a DataFrame with no instantiations, and a Series with the instantations."""
    #    assert len(f) == 1
