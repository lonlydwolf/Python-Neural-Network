"""Historical training entry point, which writes a per-sample output trace."""

import argparse
from pathlib import Path

from Classes.Trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_dir",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent / "Data",
        help="Folder containing inputs.txt; outputs.txt in this folder is replaced.",
    )
    args = parser.parse_args()
    print(args.data_dir.resolve())
    Trainer(args.data_dir).Train()


if __name__ == "__main__":
    main()
