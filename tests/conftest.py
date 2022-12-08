import glob
import os
import pytest
import logging
import pandas as pd

TEST_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.dirname(TEST_DIR))

# some files for testing
DOG_FILE = os.path.join(ROOT_DIR, "testing/dog_problem.BIFXML")
LEC1_FILE = os.path.join(ROOT_DIR, "testing/lecture_example.BIFXML")
LEC2_FILE = os.path.join(ROOT_DIR, "testing/lecture_example2.BIFXML")


def pytest_sessionstart(session):
    """runs before all tests start https://stackoverflow.com/a/35394239"""
    print("in sessionstart")

    debug = os.environ.get("DEBUG", "0") == "1"
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=log_level)
    logging.info(f"setup logging with debug level = {debug}")


@pytest.fixture(autouse=True)
def run_around_tests():
    """code to run before and afer each test https://stackoverflow.com/a/62784688/5500073"""
    # code that will run before a given test:

    yield
    # code that will run after a given test:


def compare_frames(f1: pd.DataFrame, f2: pd.DataFrame):
    """
    Assert two dataframes are equivalent.
    Created this because f1.equals(f2) seems to incorrectly return False for marginalization tests.
    """
    same = False

    same = type(f1) == type(f2) and type(f1) == pd.DataFrame

    same = same and f1.columns.values.tolist() == f2.columns.values.tolist()
    same = same and f1.index.values.tolist() == f2.index.values.tolist()
    same = same and set((f1 == f2).all(True)) == set([True])

    if not same:
        print("dataframes are different:")
        print(f1)
        print()
        print(f2)
    assert same
