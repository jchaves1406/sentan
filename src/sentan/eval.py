from abc import ABC, abstractmethod
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from pandas import DataFrame, Series
from typing import List, Dict

class AbstractMetric(ABC):
    model: Pipeline
     
    def set_model(self, model: Pipeline) -> "AbstractMetric":
        self.model = model
        return self

    @abstractmethod
    def call(self, data: Series, label: Series) -> float:
        pass

class AccuracyMetric(AbstractMetric):
    def call(self, data: Series, label: Series) -> float:
        y_pred = self.model.predict(data)
        return accuracy_score(label, y_pred)

class ModelEvaluator:
    def __init__(self, metrics: Dict[str, AbstractMetric]):
        self.metrics = metrics

    def evaluate(self, data: Series, label: Series, model: Pipeline) -> float:
        for name, metric in self.metrics.items():
            result = metric.set_model(model).call(data, label)
            print(f"{name}: {result:.4f}")