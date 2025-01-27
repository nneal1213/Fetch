import torch
from torch import nn
from transformers import AutoTokenizer, DistilBertModel

from models import MultiTaskEmbeddingModel

torch.manual_seed(0)


MODEL_NAME: str = "distilbert-base-uncased"
NUM_CATEGORIES: int = 5
sentences = ["Hello, my dog is cute"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
huggingface_model = DistilBertModel.from_pretrained(MODEL_NAME)

token_inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)

task_a_head = nn.Sequential(nn.Linear(huggingface_model.config.hidden_size, NUM_CATEGORIES),
                            nn.LogSoftmax(dim=1)  # used for numerical stability
                            )

task_b_head = nn.Sequential(nn.Linear(huggingface_model.config.hidden_size, 1),
                            nn.Sigmoid()
                            )

model = MultiTaskEmbeddingModel(model=huggingface_model, task_heads=(task_a_head, task_b_head))


def main():
    with torch.no_grad():
        output_a, output_b = model(token_inputs)

    output_a = torch.exp(output_a)

    pythonic_output_a = output_a.flatten().tolist()
    pythonic_output_b = output_b.item()

    print(f"Output - Multi-categorical Classification: {list(map(lambda x: float('{:.3f}'.format(x)), pythonic_output_a))}")
    print(f"Output - Sentiment Analysis: {pythonic_output_b:.2f}")

    assert isinstance(pythonic_output_a, list)
    assert (1 - 1E-6) < sum(pythonic_output_a) < (1 + 1E-6)

    assert isinstance(pythonic_output_b, float)
    assert 0.0 < pythonic_output_b < 1.0


if __name__ == "__main__":
    main()
