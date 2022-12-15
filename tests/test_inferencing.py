from BNReasoner import BNReasoner, Ordering
from BayesNet import BayesNet
import pandas as pd
from pandas.testing import assert_frame_equal
from tests.conftest import LEC2_FILE
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


def test_marginal_dist__posterior():
    """
    This test corresponds to slide 7-9 Lecture 4.
    """
    bn = BayesNet()
    bn.create_bn(
        ["A", "B", "C"],
        [("A", "B"), ("B", "C")],
        {
            "A": copy.deepcopy(df_a),
            "B": copy.deepcopy(df_b),
            "C": copy.deepcopy(df_c),
        },
    )
    br = BNReasoner(bn)
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


def test_MPE():
    """This test corresponds to slide 'Example 1 - MPE'."""
    br = BNReasoner(LEC2_FILE)
    e = pd.Series({"J": True, "O": False})
    assignments, prob = br.MPE(e, ordering_method=Ordering.MIN_DEG)

    expected = pd.Series({"I": False, "J": True, "O": False, "X": False, "Y": False})
    assert_frame_equal(expected, assignments)
    assert prob == 0.2304225

    # should get same result regardless of ordering method
    assignments, prob = br.MPE(e, ordering_method=Ordering.MIN_FILL)
    assert_frame_equal(expected, assignments)
    assert prob == 0.2304225

    # TODO: consider mocking/hacking the ordering function to return the same order as in the slides...
    # and test this out piece by piece...

    # TODO: add test from workgroup?


def test_MAP():
    """This test corresponds to slide 'Example 2 - Compute MAP'."""
    br = BNReasoner(LEC2_FILE)
    e = pd.Series({"O": True})
    Q = {"I", "J"}

    # import pdb; pdb.set_trace()
    prob, asn = br.MAP(Q, e, ordering_method=Ordering.MIN_DEG)
    expected = pd.Series({"I": True, "J": False})
    assert_frame_equal(expected, asn)
    assert prob == 0.242720

    # should get same result regardless of ordering method
    prob, asn = br.MAP(Q, e, ordering_method=Ordering.MIN_FILL)
    assert_frame_equal(expected, asn)
    assert prob == 0.242720

    # old map test
    ''' prob = br.MAP(Q, e, ordering_method=Ordering.MIN_DEG)
    expected = pd.DataFrame([0.242720], columns=['p'])
    assert_frame_equal(expected, prob)'''

    # TODO: add test from workgroup?


def test_variable_elim__simple():
    """Taken from slide 18 Lecture 3."""

    bn = BayesNet()
    bn.create_bn(
        ["A", "B", "C"],
        [("A", "B"), ("B", "C")],
        {
            "A": copy.deepcopy(df_a),
            "B": copy.deepcopy(df_b),
            "C": copy.deepcopy(df_c),
        },
    )
    br = BNReasoner(bn)
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
