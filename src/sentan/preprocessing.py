from abc import ABC, abstractmethod
from pandas import DataFrame
from typing import List
from unidecode import unidecode
import re

class AbstractPreprocessor(ABC):
    data: DataFrame
    clean_data: DataFrame

    def set_data(self, data: DataFrame) -> "AbstractPreprocessor":
        self.data = data

    def get_data(self) -> DataFrame:
        return self.clean_data

    @abstractmethod
    def preprocess(self) -> "AbstractPreprocessor":
        pass


class AbstractTextStep(ABC):
    def call(self, text: str) -> str:
        return text
    

class DupSpacesStep(AbstractTextStep):
    def __init__(self):
        self.pattern = re.compile(r"\s+")

    def call(self, text: str) -> str:
        return re.sub(self.pattern, " ", text)


class LowerStep(AbstractTextStep):
    def call(self, text: str) -> str:
        return text.lower()
    

class UnicodeStep(AbstractTextStep):
    def call(self, text: str) -> str:
        return unidecode(text)


class TextPreprocessor(AbstractPreprocessor):
    def __init__(self, column: str, steps: List[AbstractTextStep]):
        self.column = column
        self.steps = steps

    def preprocess(self):
        def clean_function(text: str) -> str:
            for step in self.steps:
                text = step.call(text)
            return text
        self.clean_data = self.data.assign(**{self.column: lambda df: df[self.column].apply(clean_function)})
        return self