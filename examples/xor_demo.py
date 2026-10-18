"""Train the original Python network and evaluate all four XOR inputs."""

import argparse
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Classes.DataHandler import DataHandler
from Classes.Network import Network


DATA_DIR = ROOT / "Data"
XOR_CASES = ((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0))


def train_xor(seed=0):
    """Use a fixed initialization seed and the checked-in training sequence."""
    data = DataHandler(DATA_DIR, write_output=False)
    data.read()
    random.seed(seed)
    network = Network(data.Topology)
    for _ in range(data.NumberOfTestCases):
        network.feedForward(data.nextInputs())
        network.backPropagation(data.nextTargets())
    return network


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="Weight initialization seed (default: 0).")
    args = parser.parse_args()
    network = train_xor(args.seed)
    print(f"Seed: {args.seed}; training examples: 4000; topology: 2-4-1")
    print("Inputs  Target  Prediction  Class (threshold 0.5)")
    correct = 0
    for left, right, target in XOR_CASES:
        network.feedForward([left, right])
        prediction = network.getResult()[0]
        predicted_class = int(prediction >= 0.5)
        correct += predicted_class == target
        print(f"{left} {right}     {target}       {prediction:.6f}    {predicted_class}")
    print(f"Correct: {correct}/4")


if __name__ == "__main__":
    main()
