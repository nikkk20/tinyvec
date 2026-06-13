from sentence_transformers import SentenceTransformer
import numpy as np



model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "cat",
    "cat"
    # "The cat sat on the mat",
    # "A dog is running in the park",
    # "The weather is sunny today",
    # "Quantum computers use qubits",
    # "The kitten is sleeping on the rug"
]

embeddings = model.encode(sentences)
print(type(embeddings))
print(embeddings.shape) 
print(embeddings[:2]) 
