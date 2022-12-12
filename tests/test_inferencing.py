from BNReasoner import BNReasoner, Ordering
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
