from BNReasoner import BNReasoner, Ordering
import pandas as pd
from pandas.testing import assert_frame_equal
from tests.conftest import LEC2_FILE


def test_MEP():
    """This test corresponds to slide 'Example 1 - MPE'."""
    br = BNReasoner(LEC2_FILE)
    e = pd.series({"J": True, "O": False})
    assignments, prob = br.MPE(e, ordering_method=Ordering.MIN_DEG)

    expected = pd.series({"I": False, "J": True, "O": False, "X": False, "Y": False})
    assert_frame_equal(expected, assignments)
    assert prob == 0.2304225

    # should get same result regardless of ordering method
    assignments, prob = br.MPE(e, ordering_method=Ordering.MIN_FILL)
    assert_frame_equal(expected, assignments)
    assert prob == 0.2304225


def test_MAP():
    """This test corresponds to slide 'Example 2 - Compute MAP'."""
    br = BNReasoner(LEC2_FILE)
    e = pd.series({"O": True})
    Q = {"I", "J"}

    assignments, prob = br.MAP(Q, e, ordering_method=Ordering.MIN_DEG)
    expected = pd.series({"I": True, "J": False})
    assert_frame_equal(expected, assignments)
    assert prob == 0.242720

    # should get same result regardless of ordering method
    assignments, prob = br.MAP(Q, e, ordering_method=Ordering.MIN_FILL)
    assert_frame_equal(expected, assignments)
    assert prob == 0.242720
