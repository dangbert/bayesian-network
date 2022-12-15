from tests.conftest import DOG_FILE, LEC1_FILE, LEC2_FILE, compare_frames
from BNReasoner import BNReasoner, Ordering
from BayesNet import BayesNet
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
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

    # TODO: add some "harder" examples on other networks


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
            "B": [False, True],
            "p": [0.06 + 0.32, 0.54 + 0.08],
        }
    )
    compare_frames(res, expected)

    res = BNReasoner.marginalize(cpt, "B")
    expected = pd.DataFrame(
        {
            "A": [False, True],
            "p": [0.08 + 0.32, 0.54 + 0.06],
        }
    )
    compare_frames(res, expected)

    cpt = copy.deepcopy(FACTOR_EX3)
    res = BNReasoner.marginalize(cpt, "D")
    expected = pd.DataFrame(
        {
            "B": [False, False, True, True],
            "C": [False, True, False, True],
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
    # TEST 1
    cpt = copy.deepcopy(FACTOR_EX3)
    res, assignments = BNReasoner.max_out(cpt, "D")
    expected_res = pd.DataFrame(
        {
            "B": [False, False, True, True],
            "C": [False, True, False, True],
            "p": [1.0, 0.8, 0.9, 0.95],
        }
    )
    expected_assignments = pd.Series([False, True, True, True], name='D')
    assert_frame_equal(res, expected_res)
    assert_series_equal(assignments, expected_assignments)

    # TEST 2: does not pass!! since assignment is wrong!
    cpt = copy.deepcopy(FACTOR_EX1)
    res, asn = BNReasoner.max_out(cpt, "hear-bark")
    expected_res = pd.DataFrame(
        {
            "dog-out": [False, True],
            "p": [0.7, 0.99],
        }
    )
    expected_asn = pd.Series([False, True], name='hear-bark')
    assert_frame_equal(res, expected_res)
    assert_series_equal(asn, expected_asn)


    '''# TEST 3
    cpt = copy.deepcopy(FACTOR_EX1)
    res, assignments = BNReasoner.max_out(cpt, "dog-out")
    expected_res = pd.DataFrame(
        {
            "hear-bark": [False, True],
            "p": [0.7, 0.99],
        }
    )

    # import pdb; pdb.set_trace()

    expected_assignments = pd.Series([False, True], name='dog-out')
    assert_frame_equal(res, expected_res)
    assert_series_equal(assignments, expected_assignments)'''


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
    assert res.shape == (16, 5)
    assert res.columns.values.tolist() == ["B", "C", "D", "E", "p"]

    # for simplicity we'll just check that certain expected rows exist
    assert (res == np.array([True, True, True, True, 0.95 * 0.448])).all(1).any()
    assert (res == np.array([True, True, True, False, 0.95 * 0.192])).all(1).any()
    assert (res == np.array([True, True, False, True, 0.05 * 0.112])).all(1).any()
    assert (res == np.array([False, False, False, False, 0.248])).all(1).any()


def test_network_pruning():
    """This test corresponds to slide 'Example 1 - MPE'."""

    br = BNReasoner(LEC2_FILE)
    orig = br.deepcopy()

    all_vars = set(br.bn.get_all_variables())

    e = pd.Series({"J": True, "O": False})

    Q = all_vars - set(dict(e).keys())

    # prune as if we're gonna do the MPE query
    br.network_pruning(Q, e)

    assert set(br.bn.get_all_variables()) == all_vars, "all vars should still exist"
    assert list(br.bn.structure.edges()) == [("I", "X"), ("Y", "O"), ("X", "O")]

    # vars for which cpts should not have changed:
    identical_cpts = {"I"}
    for var in identical_cpts:
        assert_frame_equal(br.bn.get_cpt(var), orig.bn.get_cpt(var))

    expected_new_cpts = {
        "J": pd.DataFrame(
            {
                "J": [True],
                "p": [0.5],
            }
        ),
        "Y": pd.DataFrame(
            {
                "Y": [False, True],
                "p": [0.99, 0.01],
            }
        ),
        "O": pd.DataFrame(
            {
                "Y": [False, False, True, True],
                "X": [False, True, False, True],
                "O": [False, False, False, False],
                "p": [0.98, 0.02, 0.02, 0.02],
            }
        ),
        "X": pd.DataFrame(
            {
                "I": [False, False, True, True],
                "X": [False, True, False, True],
                "p": [0.95, 0.05, 0.05, 0.95],
            }
        ),
    }
    for var, expected in expected_new_cpts.items():
        print(f"comparing cpts for var '{var}' before and after pruning")
        # (don't check dtype cause type(bool) != type(np.bool))
        # (also ignore order of columns)
        cpt = br.bn.get_cpt(var)
        assert_frame_equal(expected, cpt, check_dtype=False)
        # compare_frames(cpt, br.bn.get_cpt(var))


def test_ordering__min_deg():
    br = BNReasoner(LEC1_FILE)

    # test as if we're removing all vars
    remove_vars = {"Winter?", "Sprinkler?", "Rain?", "Wet Grass?", "Slippery Road?"}
    res = br.get_ordering(remove_vars, method=Ordering.MIN_DEG)
    assert res == ["Slippery Road?", "Wet Grass?", "Rain?", "Sprinkler?", "Winter?"]

    remove_vars = {"Sprinkler?", "Rain?"}
    res = br.get_ordering(remove_vars, method=Ordering.MIN_DEG)
    assert res == ["Sprinkler?", "Rain?"]

    remove_vars = {"Wet Grass?", "Sprinkler?", "Rain?"}
    res = br.get_ordering(remove_vars, method=Ordering.MIN_DEG)
    assert res == ["Wet Grass?", "Sprinkler?", "Rain?"]

    remove_vars = {"Rain?"}
    res = br.get_ordering(remove_vars, method=Ordering.MIN_DEG)
    assert res == ["Rain?"]

    # TODO: ideally test on another network as well or visualize ig in middle of process...


def test_ordering__min_fill():
    br = BNReasoner(LEC1_FILE)
    # int_graph = br.bn.get_interaction_graph()
    # nx.draw(int_graph, with_labels=True)
    # plt.show()

    # test as if we're removing all vars
    remove_vars = {"Winter?", "Sprinkler?", "Rain?", "Wet Grass?", "Slippery Road?"}
    res = br.get_ordering(remove_vars, method=Ordering.MIN_FILL)
    assert res == ["Slippery Road?", "Wet Grass?", "Rain?", "Sprinkler?", "Winter?"]

    # all but wet grass:
    remove_vars = {"Winter?", "Sprinkler?", "Rain?", "Slippery Road?"}
    res = br.get_ordering(remove_vars, method=Ordering.MIN_FILL)
    assert res == ["Slippery Road?", "Winter?", "Rain?", "Sprinkler?"]

    # (both vars add 0 new interactions)
    remove_vars = {"Wet Grass?", "Winter?"}
    res = br.get_ordering(remove_vars, method=Ordering.MIN_FILL)
    assert res == ["Wet Grass?", "Winter?"]

    remove_vars = {"Rain?"}
    res = br.get_ordering(remove_vars, method=Ordering.MIN_FILL)
    assert res == ["Rain?"]
