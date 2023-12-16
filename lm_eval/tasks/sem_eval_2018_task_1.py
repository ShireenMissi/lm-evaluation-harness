
from lm_eval.base import MultipleChoiceTask
from lm_eval.metrics import mean
import numpy as np
class SemEval2018Task1(MultipleChoiceTask):
    VERSION = 0

    DATASET_PATH = "sem_eval_2018_task_1"
    DATASET_NAME = "subtask5.english"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(
                    map(self._process_doc, self.dataset["train"])
                )
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        if self.has_test_docs():
            return map(self._process_doc, self.dataset["test"])



    def _process_doc(self, doc):
        return {
            "query": self.preprocess(doc["Tweet"]),  # The query prompt.
            "choices": self.choices(),  # The list of choices.
            "gold": self.gold_choices(doc),  # The integers used to index into the correct elements of `"choices"`.
        }

    def preprocess(self, text):
        emotions = "anger, anticipation, disgust, fear, joy, love, optimism, pessimism, sadness, surprise, and trust"
        return f"Now I want you to perform a classification of the following sentence based on the emotion it represents, you can use {emotions}. {text}"

    def gold_choices(self, doc):
        return [i for i, choice in enumerate(self.choices()) if doc[choice]]

    def choices(self):
        return['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust']

    def doc_to_text(self, doc):
        return doc["query"]

    def aggregation(self):

        return {"acc": mean,"acc_norm": mean}

    def process_results(self, doc, results):
        gold = doc["gold"]

        predicted = np.argmax(results)

        acc = 1.0 if predicted in gold else 0.0
        completion_len = np.array([float(len(i)) for i in doc["choices"]])
        acc_norm = 1.0 if np.argmax(results / completion_len) in gold else 0.0

        return {
            "acc": acc,
            "acc_norm": acc_norm
        }

    def doc_to_target(self, doc):
        return " ".join([doc["choices"][index] for index in doc["gold"]])

