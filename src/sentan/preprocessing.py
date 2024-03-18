from abc import ABC, abstractmethod
from pandas import DataFrame
from typing import List
from unidecode import unidecode
from nltk import word_tokenize
from nltk.corpus import stopwords
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
    

class LowerStep(AbstractTextStep):
    def call(self, text: str) -> str:
        return text.lower()
    

class UnicodeStep(AbstractTextStep):
    def call(self, text: str) -> str:
        return unidecode(text)


class WordTokenStep(AbstractTextStep):
    def __init__(self, lang: str = "english"):
        self.lang = lang

    def call(self, text: str) -> str:
        return " ".join(word_tokenize(text, language=self.lang))


class StopWordsStep(AbstractTextStep):
    def __init__(self, lang: str = "english"):
        self.lang = lang
        self.stop_words = stopwords.words(self.lang)

    def call(self, text: str) -> str:
        tokens = text.split(" ")
        filtered_tokens = filter(lambda token: token not in self.stop_words, tokens)
        return " ".join(filtered_tokens)


class LenFilterStep(AbstractTextStep):
    def __init__(self, min_len: int, max_len: int) -> None:
        self.min_len = min_len
        self.max_len = max_len

    def call(self, text: str) -> str:
        tokens = text.split(" ")
        filtered_tokens = filter(lambda token: len(token) >= self.min_len and len(token) <= self.max_len, tokens)
        return " ".join(filtered_tokens)


class AbstractRegexStep(AbstractTextStep, ABC):
    @abstractmethod
    def get_pattern(self) -> str:
        pass

    def call(self, text: str) -> str:
        pattern = self.get_pattern()
        return re.sub(pattern, " ", text)


class URLRemovalStep(AbstractRegexStep):
    def get_pattern(self) -> re.Pattern:
        return re.compile(r"https?://[^\s]+ ")


class SPRemovalStep(AbstractRegexStep):
    def get_pattern(self) -> re.Pattern:
        return re.compile(r"[^a-zA-Z\s]+")
    

class DupSpacesStep(AbstractTextStep):
    def get_pattern(self) -> re.Pattern:
        return re.compile(r"\s+")


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