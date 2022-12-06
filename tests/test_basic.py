from tests.conftest import DOG_FILE
from BNReasoner import BNReasoner
import pandas as pd


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
    cpt = pd.DataFrame(
        {
            "dog-out": [False, False, True, True],
            "hear-bark": [False, True, False, True],
            "p": [0.7, 0.3, 0.01, 0.99],
        }
    )
    # assert cpt.equals(br.bn.get_cpt("hear-bark"))
    # print(cpt)

    new_cpt = BNReasoner.marginalize(cpt, "dog-out")
    assert new_cpt.equals(
        pd.DataFrame(
            {
                "hear-bark": [False, True],
                "p": [0.70 + 0.01, 0.99 + 0.3],
            }
        )
    )

    # another test (from slide 90 of bayesian_combined.pdf)
    cpt = pd.DataFrame(
        {
            "A": [True, True, False, False],
            "B": [True, False, True, False],
            "p": [0.54, 0.06, 0.08, 0.32],
        }
    )
    assert BNReasoner.marginalize(cpt, "A").equals(
        pd.DataFrame(
            {
                "B": [True, False],
                "p": [0.54 + 0.08, 0.06 + 0.32],
            }
        )
    )
    assert BNReasoner.marginalize(cpt, "B").equals(
        pd.DataFrame(
            {
                "A": [True, False],
                "p": [0.54 + 0.06, 0.08 + 0.32],
            }
        )
    )

    # more complicated test with 3 vars...
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
