from abc import ABC, abstractmethod
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sentan.eval import ModelEvaluator
from typing import Dict

class AbstractModelBuilder(ABC):
    model: Pipeline

    def get_model(self) -> Pipeline:
        return self.model

    @abstractmethod
    def build(self):
        pass


class TFIDFLogisticBuilder(AbstractModelBuilder):
    def build(self) -> "TFIDFLogisticBuilder":
        self.model = Pipeline([
            ("vect", CountVectorizer()),
            ("clf", LogisticRegression())
        ])


class AbstractModelProccessor(ABC):
    model: Pipeline
    data: DataFrame

    def set_elements(self, model: Pipeline, data: DataFrame) -> "AbstractModelProccessor":
        self.model = model
        self.data = data
        return self

    def get_model(self) -> Pipeline:
        return self.model

    @abstractmethod
    def process(self) -> "AbstractModelProccessor":
        pass

class TrainModelProccessor(AbstractModelProccessor):
    def __init__(
        self, 
        text_column: str, 
        label_column: str, 
        partition_column: str, 
        evaluator: ModelEvaluator, 
        label_maps: Dict[str, int]
    ) -> None:
        self.text_column = text_column
        self.label_column = label_column
        self.partition_column = partition_column
        self.evaluator = evaluator
        self.label_maps = label_maps


    def process(self) -> "TrainModelProccessor":
        self.data = self.data.assign(
            label=self.data[self.label_column].map(self.label_maps)
        )
        train_data = self.data.query(f"{self.partition_column} == 'train'")
        test_data = self.data.query(f"{self.partition_column} == 'test'")

        train_corpus, train_labels = train_data[self.text_column], train_data[self.label_column]
        test_corpus, test_labels = test_data[self.text_column], test_data[self.label_column]

        self.model.fit(train_corpus, train_labels)
        self.evaluator.evaluate(test_corpus, test_labels, self.model)
        return self