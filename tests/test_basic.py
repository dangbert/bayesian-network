from tests.conftest import DOG_FILE, compare_frames
from BNReasoner import BNReasoner
import pandas as pd
from pandas.testing import assert_frame_equal
import copy
import numpy as np


def test_d_separation():

    br = BNReasoner(DOG_FILE)

    res = br.d_separated(set(["bowel-problem"]), set(["light-on"]), set(["hear-bark"]))
    assert res == False

    # check fork 'family-out'
    res = br.d_separated(
        set(["bowel-problem"]), set(["light-on"]), set(["hear-bark", "family-out"])
    )
    assert res == True
    res = br.d_separated(set(["bowel-problem"]), set(["light-on"]), set(["family-out"]))
    assert res == True

    # check collider 'dog-out'
    res = br.d_separated(set(["bowel-problem"]), set(["light-on"]), set([]))
    assert res == True


FACTOR_EX1 = pd.DataFrame(
    {
        "dog-out": [False, False, True, True],
        "hear-bark": [False, True, False, True],
        "p": [0.7, 0.3, 0.01, 0.99],
    }
)

# from slide 90 of bayesian_combined.pdf
FACTOR_EX2 = pd.DataFrame(
    {
        "A": [True, True, False, False],
        "B": [True, False, True, False],
        "p": [0.54, 0.06, 0.08, 0.32],
    }
)

# from slide 115 "Maximising-Out - Introduction":
FACTOR_EX3 = pd.DataFrame(
    {
        "B": [True, True, True, True, False, False, False, False],
        "C": [True, True, False, False, True, True, False, False],
        "D": [True, False, True, False, True, False, True, False],
        "p": [0.95, 0.05, 0.9, 0.1, 0.8, 0.2, 0.0, 1.0],
    }
)


def test_marginalization():
    # br = BNReasoner(DOG_FILE)

    # note: cpt == br.bn.get_cpt("hear-bark")
    """
        dog-out  hear-bark     p
    0    False      False  0.70
    1    False       True  0.30
    2     True      False  0.01
    3     True       True  0.99
    """
    cpt = copy.deepcopy(FACTOR_EX1)
    # assert cpt.equals(br.bn.get_cpt("hear-bark"))
    # print(cpt)

    res = BNReasoner.marginalize(cpt, "dog-out")
    expected = pd.DataFrame(
        {
            "hear-bark": [False, True],
            "p": [0.70 + 0.01, 0.99 + 0.3],
        }
    )
    compare_frames(res, expected)
    # assert_frame_equal(res, expected) # note this fails due to a dtype difference

    cpt = copy.deepcopy(FACTOR_EX2)
    res = BNReasoner.marginalize(cpt, "A")
    expected = pd.DataFrame(
        {
            "B": [True, False],
            "p": [0.54 + 0.08, 0.06 + 0.32],
        }
    )
    compare_frames(res, expected)

    res = BNReasoner.marginalize(cpt, "B")
    expected = pd.DataFrame(
        {
            "A": [True, False],
            "p": [0.54 + 0.06, 0.08 + 0.32],
        }
    )
    compare_frames(res, expected)

    cpt = copy.deepcopy(FACTOR_EX3)
    res = BNReasoner.marginalize(cpt, "D")
    expected = pd.DataFrame(
        {
            "B": [True, True, False, False],
            "C": [True, False, True, False],
            "p": [1.0, 1.0, 1.0, 1.0],
        }
    )
    compare_frames(res, expected)

    # TODO: more complicated test with 3 vars...
    # cpt = pd.DataFrame(
    #    {
    #        "A": [True, True, True, True, True, True, True, True],
    #        "B": [True, True, False, False, True, True, False, False],
    #        "E": [True, False, True, False, True, False, True, False],
    #        "p": [0.95, 0.94, 0.29, 0.011, 0.05, 0.06, 0.71, 0.999],
    #    }
    # )
    # cpt_be = BNReasoner.marginalize(cpt, "A")
    #    {
    #        "B": [True, True, ],
    #        "E": [True, ],
    #        "p": [0.95, 0.94, 0.29, 0.011, 0.05, 0.06, 0.71, 0.999],
    #    }


# res = br.marginalize()
# assert res == False


def test_maxing_out():
    cpt = copy.deepcopy(FACTOR_EX3)
    res = BNReasoner.max_out(cpt, "D")
    expected = pd.DataFrame(
        {
            "B": [True, True, False, False],
            "C": [True, False, True, False],
            "p": [0.95, 0.9, 0.8, 1.0],
        }
    )
    assert res.equals(expected)

    res = BNReasoner.max_out(cpt, "D")
    cpt = copy.deepcopy(FACTOR_EX1)

    expected = pd.DataFrame(
        {
            "dog-out": [False, True],
            "p": [0.7, 0.99],
        }
    )
    res = BNReasoner.max_out(cpt, "hear-bark")
    assert res.equals(expected)

    expected = pd.DataFrame(
        {
            "hear-bark": [False, True],
            "p": [0.7, 0.99],
        }
    )
    res = BNReasoner.max_out(cpt, "dog-out")
    assert res.equals(expected)


def test_multiply_factors():
    # from "multiplication of factors" slides:
    f1 = copy.deepcopy(FACTOR_EX3)
    f2 = pd.DataFrame(
        {
            "D": [True, True, False, False],
            "E": [True, False, True, False],
            "p": [0.448, 0.192, 0.112, 0.248],
        }
    )
    res = BNReasoner.multiply_factors(f1, f2)
    assert res.shape == (32, 4)
    # assert set(res.columns.values.tolist()) == set(['B', 'C', 'D', 'E', 'p'])
    assert res.columns.values.tolist() == ["B", "C", "D", "E", "p"]

    # for simplicity we'll just check that certain expected rows exist
    assert (res == np.array([True, True, True, True, 0.95 * 0.448])).all(1).any()
    assert (res == np.array([True, True, True, False, 0.95 * 0.192])).all(1).any()
    assert (res == np.array([True, True, False, True, 0.05 * 0.112])).all(1).any()
    assert (res == np.array([False, False, False, False, 0.248])).all(1).any()

    # TODO: can we test an example when f2 has more than 2 variables??


def test_network_pruning():

    Q = {"C", "B"}
    e = {"D": "True"}

    res = BNReasoner.network_pruning(Q, e)

    expected = pd.DataFrame(
        {
            "B": [True, True, False, False],
            "C": [True, False, True, False],
            "D": [True, True, True, True],
            "p": [0.95, 0.9, 0.8, 1.0],
        }
    )

    assert res.equals(expected)
