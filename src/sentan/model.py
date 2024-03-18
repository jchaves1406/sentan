from abc import ABC, abstractmethod
from pandas import DataFrame
from sklearn.pipeline import Pipeline

class AbstractModelBuilder(ABC):
    model: Pipeline

    def get_model(self) -> Pipeline:
        return self.model

    @abstractmethod
    def build(self):
        pass

class AbstractModelProccessor(ABC):
    model: Pipeline
    data: DataFrame

    def set_elements(self, model: Pipeline, data: DataFrame) -> "AbstractModelProccessor":
        self.model = model
        self.data = data

    def get_model(self) -> Pipeline:
        return self.model

    @abstractmethod
    def process(self) -> "AbstractModelProccessor":
        pass