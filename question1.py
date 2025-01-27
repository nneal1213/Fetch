import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from torch import nn

from models import SimpleEmbeddingModel

SentencePair = tuple[str, str]
EMBEDDING_MODEL_NAME = "paraphrase-MiniLM-L3-v2"


def main(sentences: SentencePair, model: nn.Module):
    with torch.no_grad():
        embedding1, embedding2 = model(sentences)
    cosine_similarity = util.cos_sim(embedding1, embedding2).item()
    print(f'Sentences: {sentences}')
    print(f'Cosine similarity of sentences: {cosine_similarity:.2f}')
    print(f'Length of Embeddings {len(embedding1)}, {len(embedding2)}\n')

    return cosine_similarity


if __name__ == "__main__":
    huggingface_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    model = SimpleEmbeddingModel(embedding_model=huggingface_model)
    similar_sentences: SentencePair = ('this is a tangerine', 'this is an orange')
    different_sentences: SentencePair = ('A man is on a tree', 'a dog is in the car')

    print('Executing similar sentences..')
    cosine_similarity_similar_sentences = main(similar_sentences, model)

    print('Executing different sentences..')
    cosine_similarity_different_sentences = main(different_sentences, model)

    # sanity check
    assert cosine_similarity_similar_sentences > cosine_similarity_different_sentences
    print('Cosine similarity check passed!')
