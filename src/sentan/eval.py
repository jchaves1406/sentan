from abc import ABC, abstractmethod
from sklearn.pipeline import Pipeline
from pandas import DataFrame
from typing import List

class AbstractMetric(ABC):
    model: Pipeline
     
    def set_model(self, model: Pipeline) -> "AbstractMetric":
        self.model = model
        return self

    @abstractmethod
    def call(self, data: DataFrame) -> float:
        pass

class ModelEvaluator:
    def __init__(self, metric: List[AbstractMetric], model: Pipeline):
        self.metric = metric
        self.model = model

    def evaluate(self, data: DataFrame) -> float:
        for metric in self.metric:
            response = metric.call(data)
            print(f"{metric.__class__.__name__}: {response}")