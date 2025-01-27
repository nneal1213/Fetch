from torch import nn
import torch
from sentence_transformers import SentenceTransformer
from collections.abc import Iterable
from transformers import AutoTokenizer, DistilBertModel
from transformers.utils.dummy_pt_objects import PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding

TaskHeadList = Iterable[nn.Sequential]

class SimpleEmbeddingModel(nn.Module):
    """
    A simple embedding model
    """

    def __init__(self, embedding_model: SentenceTransformer):
        super(SimpleEmbeddingModel, self).__init__()
        self.model = embedding_model

    def forward(self, sentence: str | Iterable[str]):
        return self.model.encode(sentence)



class MultiTaskEmbeddingModel(nn.Module):
    """
    A multi-task embedding model
    Supports two tasks:
    - Task A: Sentence classification into `num_classes_task_a` classes.
    - Task B: Sentiment Analysis
    """

    def __init__(self, model: PreTrainedModel, task_heads: TaskHeadList):
        super(MultiTaskEmbeddingModel, self).__init__()
        self.model = model
        self.task_heads = task_heads


    def forward(self, token_inputs: BatchEncoding) -> list[torch.Tensor]:
        """
        Forward pass for the multi-task model.

        Args:
            sentences: A list of input sentences.

        Returns:
            A list containing outputs of task heads, defined via dependency injection"""

        device = next(self.parameters()).device
        token_inputs = {k: v.to(device) for k, v in token_inputs.items()}

        outputs = self.model(**token_inputs)
        classification_outputs = outputs.last_hidden_state[:, 0, :]

        outputs = [task_head(classification_outputs) for task_head in self.task_heads]
        return outputs
