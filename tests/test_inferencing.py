from BNReasoner import BNReasoner, Ordering
from BayesNet import BayesNet
import os
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from numpy.testing import assert_approx_equal
from examples import LEC2_FILE, USE_CASE_FILE
from tests.conftest import ROOT_DIR

import copy

df_a = pd.DataFrame(
    {
        "A": [True, False],
        "p": [0.6, 0.4],
    }
)
df_b = pd.DataFrame(
    {
        "A": [True, True, False, False],
        "B": [True, False, True, False],
        "p": [0.9, 0.1, 0.2, 0.8],
    }
)
df_c = pd.DataFrame(
    {
        "B": [True, True, False, False],
        "C": [True, False, True, False],
        "p": [0.3, 0.7, 0.5, 0.5],
    }
)

BN_ABC = BayesNet()
BN_ABC.create_bn(
    ["A", "B", "C"],
    [("A", "B"), ("B", "C")],
    {
        "A": copy.deepcopy(df_a),
        "B": copy.deepcopy(df_b),
        "C": copy.deepcopy(df_c),
    },
)


def test_marginal_dist__posterior():
    """
    This test corresponds to slide 7-9 Lecture 4.
    """
    br = BNReasoner(copy.deepcopy(BN_ABC))
    Q = {"C"}
    e = pd.Series({"A": True})

    res = br.marginal_distribution(Q, e)

    expected = pd.DataFrame(
        {
            "C": [False, True],
            "p": [0.68, 0.32],
        }
    )
    assert_frame_equal(expected, res, check_dtype=False)


def test_marginal_dist__prior():
    """
    Test marginal_dist when evidence is empty
    (i.e. get "prior marginal" prob of B).
    See slides on "Variable Elimination"
    """
    br = BNReasoner(copy.deepcopy(BN_ABC))
    res = br.marginal_distribution({"C"}, pd.Series({}))
    expected = pd.DataFrame(
        {
            "C": [False, True],
            "p": [0.624, 0.376],
        }
    )
    assert_frame_equal(expected, res, check_dtype=False)

    br = BNReasoner(copy.deepcopy(BN_ABC))
    res = br.marginal_distribution({"B"}, pd.Series({}))
    expected = pd.DataFrame(
        {
            "B": [False, True],
            "p": [0.38, 0.62],
        }
    )
    assert_frame_equal(expected, res, check_dtype=False)


def test_marginal_dist__hard():
    """Test marginal dist behaves the same before and after network pruning!"""
    bif_file = os.path.join(ROOT_DIR, "experiments/dataset/random00.bifxml")
    assert os.path.isfile(bif_file)
    orig = BNReasoner(bif_file)

    query = {"Q": ["7"], "E": {"2": False, "0": True}}
    Q, e = set(query["Q"]), pd.Series(query["E"])

    from examples import visualize

    expected = orig.deepcopy().marginal_distribution(
        Q, e, ordering_method=Ordering.MIN_FILL
    )

    # try pruning first!!
    for i in range(5):
        br = orig.deepcopy()
        br.network_pruning(Q, e)
        res = br.marginal_distribution(Q, e, ordering_method=Ordering.MIN_FILL)
        assert_frame_equal(res, expected)


"""
def test_marginal_dist__use_case():
    br = BNReasoner(USE_CASE_FILE)
    expected = br.bn.get_cpt("woman")

    # 'woman' is a root node so its prior probability should just be its CPT
    res = br.marginal_distribution(
        {"woman"},
        pd.Series({}),
        ordering_method=Ordering.MIN_FILL,
    )
    assert_frame_equal(expected, res, check_dtype=False)
"""


def test_MPE():
    """This test corresponds to slide 'Example 1 - MPE'."""
    br = BNReasoner(LEC2_FILE)
    e = pd.Series({"J": True, "O": False})
    prob, asn = br.MPE(e, ordering_method=Ordering.MIN_DEG)

    expected = pd.Series({"I": False, "J": True, "O": False, "X": False, "Y": False})
    assert_approx_equal(prob, 0.2304225)
    assert_series_equal(expected, asn)

    # should get same result regardless of ordering method
    prob, asn = br.MPE(e, ordering_method=Ordering.MIN_FILL)
    assert_approx_equal(prob, 0.2304225)
    assert_series_equal(expected, asn)

    # TODO: add test from workgroup?


def test_MAP():
    """This test corresponds to slide 'Example 2 - Compute MAP'."""
    br = BNReasoner(LEC2_FILE)
    e = pd.Series({"O": True})
    Q = {"I", "J"}

    prob, asn = br.MAP(Q, e, ordering_method=Ordering.MIN_DEG)
    # here we slightly differ from the slide because there's a tie of I being True or False
    #  we take I as False arbitrarily rather than True
    exp_asn = pd.Series({"I": False, "J": False})

    assert_approx_equal(prob, 0.242720)
    assert_series_equal(exp_asn, asn)

    # should get same result regardless of ordering method
    prob, asn = br.MAP(Q, e, ordering_method=Ordering.MIN_FILL)
    assert_approx_equal(prob, 0.242720)
    assert_series_equal(exp_asn, asn)

    # old map test
    """ prob = br.MAP(Q, e, ordering_method=Ordering.MIN_DEG)
    expected = pd.DataFrame([0.242720], columns=['p'])
    assert_frame_equal(expected, prob)"""

    # TODO: add test from workgroup?


def test_variable_elim__simple():
    """Taken from slide 18 Lecture 3."""
    br = BNReasoner(copy.deepcopy(BN_ABC))
    Q = {"C"}
    res = br.variable_elimination(Q)

    expected = pd.DataFrame(
        {
            "C": [False, True],
            "p": [0.624, 0.376],
        }
    )
    assert_frame_equal(expected, res, check_dtype=False)


def test_variable_elim__map():
    """Taken from slide 20 Lecture 4."""
    br = BNReasoner(LEC2_FILE)
    Q = {"I", "J"}
    E = pd.Series({"O": True})
    # to do this test properly based on the MAP slide, we have to apply the evidence first
    br._apply_evidence(E)
    res = br.variable_elimination(Q)

    expected = pd.DataFrame(
        {
            "J": [False, False, True, True],
            "I": [False, True, False, True],
            "p": [0.97088, 0.97088, 0.07712, 0.93248],
        }
    )

    assert_frame_equal(expected, res, check_dtype=False)
