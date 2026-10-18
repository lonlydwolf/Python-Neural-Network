# Python Neural Network — XOR from first principles

An educational implementation of a fully connected neural network with handwritten forward propagation, backpropagation, and momentum updates. It uses Python's standard library, without an ML framework.

This is the Python implementation linked by [XOR-First-Principles-Network](https://github.com/lonlydwolf/XOR-First-Principles-Network). It is a completed learning exercise retained as a reference, with a bounded reproducibility pass; there is no ongoing feature roadmap. It is not a general-purpose training library. The reproducible example below checks the Python implementation; it does not establish numerical parity with the C++ or C# versions.

## Run the reproducible example

Use Python 3.12 or later. No packages need to be installed. From this repository:

```bash
python3 examples/xor_demo.py --seed 0
```

The command initializes the original neural core with the selected random seed, trains once over the 4,000 checked-in examples in `Data/inputs.txt`, and prints predictions for every XOR input. It reads the dataset relative to the script, so it also works when invoked by absolute path from another directory. It does not write training logs or overwrite `Data/outputs.txt`.

For seed 0, approximate predictions are:

| Inputs | Target | Prediction |
| --- | --- | --- |
| 0, 0 | 0 | 0.00090 |
| 0, 1 | 1 | 0.97144 |
| 1, 0 | 1 | 0.97166 |
| 1, 1 | 0 | 0.00179 |

A threshold of 0.5 classifies all four correctly. Different seeds may produce different outputs; these four cases are the complete XOR truth table, not evidence of generalization to another problem. Minor floating-point differences across environments are expected.

## What the network computes

The checked-in dataset declares a `2 → 4 → 1` topology. Each non-input neuron computes `tanh(weighted_sum + 1)`. **The bias is fixed at +1 and is not trained.** Connection weights start uniformly between -0.5 and 0.5; the learning rate is 0.2 and momentum is 0.001.

For an output `a` and target `t`, the output gradient is `(t - a) * (1 - a²)`. Hidden neurons propagate that signal through outgoing weights. Each connection update is `0.2 * input * gradient + 0.001 * previous_update`. This corresponds to descending half-squared error for the single-output task. The historical training trace reports a smoothed error metric separately.

- [Network.py](Classes/Network.py): layers, forward traversal, backward traversal, and results.
- [Neuron.py](Classes/Neuron.py): activation, gradients, and weight updates.
- [DataHandler.py](Classes/DataHandler.py): the historical text dataset and optional trace output.
- [xor_demo.py](examples/xor_demo.py): deterministic training and the four-case evaluation.

## Check the implementation

From the repository root:

```bash
python3 -m unittest discover -s tests -v
```

The tests compare one initial update of all 12 weights with finite-difference gradients, verify the four XOR predictions for seeds 0–4, and run both commands from another working directory using temporary output fixtures. They check useful properties of this exercise; they do not prove correctness for every topology or seed. The neural core's equations remain unchanged by this reproducibility pass.

## Historical training trace

The original entry point remains available:

```bash
python3 main.py /path/to/data-directory
```

The selected directory must contain `inputs.txt` in the checked-in format. **This command replaces `outputs.txt` in that directory**, then records a trace for each training example. With no argument, it uses this repository's `Data` directory and replaces the tracked historical output file. Prefer `examples/xor_demo.py` for a repeatable demonstration without output-file changes.
