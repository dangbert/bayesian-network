from tests.conftest import DOG_FILE
from BNReasoner import BNReasoner


def test_d_separation():

    br = BNReasoner(DOG_FILE)

    res = br.d_seperated(set(["bowel-problem"]), set(["light-on"]), set(["hear-bark"]))
    assert res == False

    # check fork 'family-out'
    res = br.d_seperated(
        set(["bowel-problem"]), set(["light-on"]), set(["hear-bark", "family-out"])
    )
    assert res == True
    res = br.d_seperated(set(["bowel-problem"]), set(["light-on"]), set(["family-out"]))
    assert res == True

    # check collider 'dog-out'
    res = br.d_seperated(set(["bowel-problem"]), set(["light-on"]), set([]))
    assert res == True
