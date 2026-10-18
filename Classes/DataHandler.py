from pathlib import Path


class DataHandler:
    """Read the historical text dataset, with optional training-trace output."""

    def __init__(self, path, write_output=True):
        self.__DataFolderPath = Path(path)
        self.__write_output = write_output
        self.__Topology = []
        self.__numberOfTestCases = 0
        self.__passNumber = 0
        self.__Lines = []

    def read(self):
        self.__Lines = (
            self.__DataFolderPath / "inputs.txt"
        ).read_text(encoding="utf-8").splitlines(keepends=True)
        header = self.__Lines.pop(0).replace("Topology: ", "").split()
        self.__numberOfTestCases = int(header.pop())
        self.__Topology = [int(value) for value in header]
        self.__passNumber = 0
        if self.__write_output:
            (self.__DataFolderPath / "outputs.txt").write_text("", encoding="utf-8")

    def __append_output(self, text):
        if self.__write_output:
            with (self.__DataFolderPath / "outputs.txt").open(
                "a", encoding="utf-8"
            ) as output:
                output.write(text)

    def nextInputs(self):
        line = self.__Lines.pop(0)
        self.__append_output(f"Pass {self.__passNumber}:\n{line}")
        return [float(value) for value in line.replace("In: ", "").split()]

    def nextTargets(self):
        line = self.__Lines.pop(0)
        self.__append_output(line)
        self.__passNumber += 1
        return [float(value) for value in line.replace("Target: ", "").split()]

    def writeResults(self, Results, AvgError):
        results = "".join(f"{i + 1}:{result}" for i, result in enumerate(Results))
        self.__append_output(f"\nResults: {results}\nAvgError:{AvgError}\n\n")

    @property
    def Topology(self):
        return self.__Topology

    @property
    def NumberOfTestCases(self):
        return self.__numberOfTestCases
