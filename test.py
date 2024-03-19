import nltk
from pathlib import Path
from sentan.data import PartitionedCSVLoader
from sentan.preprocessing import (
    TextPreprocessor,
    AbstractTextStep,
    LowerStep,
    UnicodeStep,
    WordTokenStep,
    StopWordsStep,
    URLRemovalStep,
    SPRemovalStep,
    DupSpacesStep,
    LenFilterStep,
    )
from sentan.model import (
    BowLogisticBuilder,
    TFIDFRandomForestBuilder,
    TFIDFLogisticBuilder,
    )


print('hola')