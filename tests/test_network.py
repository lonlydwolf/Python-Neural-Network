import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Classes.Network import Network
from examples.xor_demo import XOR_CASES, train_xor


class NetworkChecks(unittest.TestCase):
    def test_weight_update_matches_numerical_gradient(self):
        random.seed(0)
        network = Network([2, 4, 1])
        # Inspect the original core without adding a testing API to it.
        connections = [
            connection
            for layer in network._Network__Layers
            for neuron in layer
            for connection in neuron.outputWeights
        ]
        original_weights = [connection.Weight for connection in connections]
        self.assertEqual(len(connections), 12)
        epsilon = 1e-6

        def loss():
            network.feedForward([1.0, 1.0])
            return 0.5 * (network.getResult()[0] - 1.0) ** 2

        numerical_gradients = []
        for connection in connections:
            weight = connection.Weight
            connection.Weight = weight + epsilon
            plus = loss()
            connection.Weight = weight - epsilon
            minus = loss()
            connection.Weight = weight
            numerical_gradients.append((plus - minus) / (2 * epsilon))

        network.feedForward([1.0, 1.0])
        network.backPropagation([1.0])
        # On the first update momentum is zero; learning rate is 0.2.
        for connection, original, numerical in zip(
            connections, original_weights, numerical_gradients
        ):
            actual_gradient = -(connection.Weight - original) / 0.2
            self.assertAlmostEqual(actual_gradient, numerical, delta=1e-8)

    def test_fixed_seed_training_learns_xor(self):
        for seed in range(5):
            with self.subTest(seed=seed):
                network = train_xor(seed)
                for left, right, target in XOR_CASES:
                    network.feedForward([left, right])
                    prediction = network.getResult()[0]
                    self.assertEqual(int(prediction >= 0.5), target)
                    self.assertLess(abs(prediction - target), 0.1)

    def test_commands_work_from_another_directory(self):
        tracked_output = ROOT / "Data" / "outputs.txt"
        original_output = tracked_output.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "fixture with spaces"
            fixture.mkdir()
            (fixture / "inputs.txt").write_text(
                "Topology: 2 4 1 4\n"
                "In: 0 In: 0\nTarget: 0\n"
                "In: 0 In: 1\nTarget: 1\n"
                "In: 1 In: 0\nTarget: 1\n"
                "In: 1 In: 1\nTarget: 0\n",
                encoding="utf-8",
            )
            env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
            subprocess.run(
                [sys.executable, str(ROOT / "main.py"), str(fixture)],
                cwd=directory, env=env, capture_output=True, text=True,
                check=True, timeout=10,
            )
            trace = (fixture / "outputs.txt").read_text(encoding="utf-8")
            self.assertEqual(trace.count("Results:"), 4)
            demo = subprocess.run(
                [sys.executable, str(ROOT / "examples" / "xor_demo.py"), "--seed", "0"],
                cwd=directory, env=env, capture_output=True, text=True,
                check=True, timeout=10,
            )
            self.assertIn("Correct: 4/4", demo.stdout)
        self.assertEqual(tracked_output.read_bytes(), original_output)


if __name__ == "__main__":
    unittest.main()
