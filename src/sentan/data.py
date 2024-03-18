import pandas as pd
from abc import ABC, abstractmethod
from pandas import DataFrame
from typing import List
from pathlib import Path

class AbstractDataLoader(ABC):
    data: DataFrame

    def get_data(self) -> DataFrame:
        return self.data

    @abstractmethod
    def load(self) -> "AbstractDataLoader":
        pass

class PartitionedCSVLoader(AbstractDataLoader):
    def __init__(self, raw_path: Path):
        self.path = raw_path

    def load(self) -> "PartitionedCSVLoader":
        files = self.raw_path.glob("*.csv")
        dfs = []
        for file in files:
            dfs.append(
                pd.read_csv(file, encoding="latin1")
                .assign(partition=file.name.split(".")[0])
            )
        self.data = (
            pd.concat(dfs)
            .filter(["text", "sentiment"])
            .dropna()
        )
        return self