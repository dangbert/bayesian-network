#!/usr/bin/env python3
"""Run an experiment, reporting stats."""
import os
import sys

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import argparse
from collections.abc import Callable
import json
from datetime import datetime
import logging
import matplotlib.pyplot as plt

from copy import deepcopy
import subprocess
import os
import time
from typing import List, Dict, Optional, Tuple
import glob

from pgmpy.readwrite import XMLBIFWriter, XMLBIFReader
from random_network import get_random_model
from BayesNet import BayesNet
from BNReasoner import BNReasoner, Ordering

from examples import visualize, USE_CASE_FILE

RAND_NETWORK_OPTIONS = {
    "n_nodes": 10,
    "edge_prob": 0.45,
    "n_states": 2,
}


"""
What's the effect of the min degree vs min fill heuristic
on the performance of variable elimination?

# generated the below using (then fixed duplicates):
# options = [str(n) for n in range(10)]
# random.choices(options, k=4)
# (lists of vars to eliminate)
"""
ELIMINATION_QUERIES = [
    ["6", "5", "0", "1"],
    ["8", "6", "3", "2"],
    ["4", "7", "8", "5"],
    ["0", "8", "2", "6"],
    ["6", "2", "7", "9"],
]

"""
# testing networking pruning on marginal dist queries.

import random
vars = [str(n) for n in range(10)]
#['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def get_one():
    Q = random.sample(vars, k=1)
    re_vars = set(vars) - set(Q)
    E_vars = random.sample(re_vars, k=2)
    E = {v: bool(random.getrandbits(1)) for v in E_vars}
    return {"Q": Q, "E": E}

MARGINAL_QUERIES = [random.choices(options, k=4) for _ in range(5)]
"""
# all of these will be run with the Ordering suggested by the results of the ELIIMINATION_QUERIES experiment
MARGINAL_QUERIES = [
    {"Q": ["4"], "E": {"8": False, "6": False}},
    {"Q": ["2"], "E": {"8": True, "4": False}},
    {"Q": ["9"], "E": {"5": False, "2": True}},
    {"Q": ["4"], "E": {"5": True, "2": False}},
    {"Q": ["7"], "E": {"2": False, "0": True}},
]


def main():
    parser = argparse.ArgumentParser(
        description="Runs and experiment, comparing different sat solver strategies."
    )

    parser.add_argument(
        "-r",
        "--replay",
        type=str,
        help="folder to reprocess stats.json (instead of running new experiment).",
    )
    parser.add_argument(
        "-m", "--message", type=str, help="description of experiment (to save)"
    )

    parser.add_argument(
        "-g",
        "--gen_networks",
        type=str,
        help="folder in which to generate random networks and exit",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.path.join(SCRIPT_DIR, "dataset/"),
        help="folder in which to load networks for experiment",
    )

    # parser.add_argument("--seed", type=int, help="random seed to use")
    args = parser.parse_args()

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(format=FORMAT, level=logging.INFO)

    if args.gen_networks:
        generate_networks(args.gen_networks)
        exit(0)

    if args.replay:
        fname = os.path.abspath(args.replay)
        if not fname.endswith(".json"):
            fname = os.path.join(fname, "stats.json")
        outdir = os.path.dirname(fname)
        assert os.path.isdir(outdir) and os.path.exists(fname)

        with open(fname, "r") as f:
            stats = json.load(f)
        logging.info(f"replaying stat processing of: '{fname}'")
        process_stats(stats, outdir)
        exit(0)

    expname = datetime.now().strftime(f"%Y_%m_%d_%H_%M_%S")
    if args.message:
        msg = args.message.replace(" ", "_")[:25]
        expname += "--" + msg
    # expname = "tmp"  # TODO for now
    # outdir = os.path.join("experiments/", expname)
    outdir = os.path.join(SCRIPT_DIR, expname)
    if not os.path.exists(outdir):
        logging.info(f"using dir: {outdir}")
        os.makedirs(outdir)

    GIT_HASH = (
        subprocess.check_output(["git", "rev-parse", "--verify", "HEAD", "--short"])
        .decode("utf-8")
        .strip()
    )
    msg = args.message if args.message else ""
    with open(os.path.join(outdir, "about.txt"), "w") as f:
        f.write(f"{msg}\n{GIT_HASH}\n")

    # run experiment
    outpath = os.path.join(outdir, "stats.json")
    nets = load_networks(args.dataset, view=False)
    assert len(nets) > 0, f"found only {len(nets)} networks in dataset"
    logging.info(f"loaded {len(nets)} networks in dataset")

    stats = run_experiment(outpath, nets)
    process_stats(stats, outdir)


def run_experiment(outpath: str, nets: List[BayesNet]) -> Dict:
    # gather list of puzzle (systems) across files
    all_stats = {"RQ1:": {}}

    def write_stats():
        nonlocal outpath
        nonlocal all_stats
        if outpath is not None:
            with open(outpath, "w") as f:
                json.dump(all_stats, f, indent=2)  # write indented json to file
                logging.info(f"wrote latest stats to: {os.path.abspath(outpath)}")

    # RQ1
    all_stats["RQ1"] = {}
    for method in [Ordering.MIN_DEG, Ordering.MIN_FILL]:
        method_stats = {"times": []}
        for net in nets:
            for vars in ELIMINATION_QUERIES:
                br = BNReasoner(deepcopy(net))
                cpu_time = time.process_time()
                all_vars = set(br.bn.get_all_variables())
                assert set(vars).issubset(all_vars)
                Q = all_vars - set(vars)
                res = br.variable_elimination(Q, method=method)
                cpu_time = time.process_time() - cpu_time
                method_stats["times"].append(cpu_time)

        all_stats["RQ1"][str(method.value)] = method_stats

    write_stats()
    return all_stats


def load_networks(dataset_dir: str, view: bool = False) -> List[BayesNet]:
    assert os.path.isdir(dataset_dir)
    pattern = os.path.join(dataset_dir, f"*.bifxml")
    logging.info(f"loading networks: '{pattern}'")

    nets = []
    for fname in glob.glob(pattern):
        net = BayesNet()
        net.load_from_bifxml(fname)
        nets.append(net)

        if view:
            visualize(net)
    return nets


def generate_networks(
    outdir: str, count: int = 10, options: Dict = RAND_NETWORK_OPTIONS
):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    assert os.path.isdir(outdir)

    for i in range(count):
        fname = os.path.join(outdir, f"random{str(i).zfill(2)}.bifxml")
        # br = get_random_br(fname, {"n_nodes": 10, "edge_prob": 0.1, "n_states": 2})

        model = get_random_model(**options)
        XMLBIFWriter(model).write_xmlbif(fname)
        logging.info(f"wrote: '{fname}'")


def process_stats(stats: Dict, outdir: str):
    logging.info(f"processing stats...\n")
    # TODO

    # graph_out = os.path.join(outdir, f"graphs.pdf")
    # plt.savefig(graph_out, dpi=400)
    # print(f"wrote: {graph_out}")


if __name__ == "__main__":
    main()