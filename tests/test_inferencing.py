from BNReasoner import BNReasoner, Ordering
from BayesNet import BayesNet
import pandas as pd
from pandas.testing import assert_frame_equal
from tests.conftest import LEC2_FILE


def test_MEP():
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

    assignments, prob = br.MAP(Q, e, ordering_method=Ordering.MIN_DEG)
    expected = pd.Series({"I": True, "J": False})
    assert_frame_equal(expected, assignments)
    assert prob == 0.242720

    # should get same result regardless of ordering method
    assignments, prob = br.MAP(Q, e, ordering_method=Ordering.MIN_FILL)
    assert_frame_equal(expected, assignments)
    assert prob == 0.242720

    # TODO: add test from workgroup?


def test_variable_elim__simple():
    """Taken from slide 18 Lecture 3."""

    bn = BayesNet()
    bn.create_bn(
        ["A", "B", "C"],
        [("A", "B"), ("B", "C")],
        {
            "A": pd.DataFrame(
                {
                    "A": [True, False],
                    "p": [0.6, 0.4],
                }
            ),
            "B": pd.DataFrame(
                {
                    "A": [True, True, False, False],
                    "B": [True, False, True, False],
                    "p": [0.9, 0.1, 0.2, 0.8],
                }
            ),
            "C": pd.DataFrame(
                {
                    "B": [True, True, False, False],
                    "C": [True, False, True, False],
                    "p": [0.3, 0.7, 0.5, 0.5],
                }
            ),
        },
    )
    br = BNReasoner(bn)
    vars = {"A", "B"}
    res = br.variable_elimination(vars)

    expected = pd.DataFrame(
        {
            "C": [True, False],
            "p": [0.376, 0.624],
        }
    )

    assert_frame_equal(expected, res, check_dtype=False)


def test_variable_elim__map():
    """Taken from slide 20 Lecture 4."""

    br = BNReasoner(LEC2_FILE)
    vars = {"O", "Y", "X"}
    res = br.variable_elimination(vars)

    expected = pd.DataFrame(
        {
            "I": [True, True, False, False],
            "J": [True, False, True, False],
            "p": [0.93248, 0.97088, 0.07712, 0.97088],
        }
    )

    assert_frame_equal(expected, res, check_dtype=False)
