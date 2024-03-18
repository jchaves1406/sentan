from abc import ABC, abstractmethod
from pandas import DataFrame
from typing import List

class AbstractDataLoader(ABC):
    data: DataFrame

    def get_data(self) -> DataFrame:
        return self.data

    @abstractmethod
    def load(self) -> "AbstractDataLoader":
        pass

