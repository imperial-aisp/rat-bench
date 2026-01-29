import numpy as np
import re
import nltk
from tqdm import tqdm
from gensim.models import Word2Vec

import gensim.downloader as api
from pandarallel import pandarallel
from nltk.tokenize import word_tokenize
from sacremoses import MosesDetokenizer
from gensim.similarities.annoy import AnnoyIndexer
import pandas as pd
from typing import List, Any

from pii_benchmark.anonymizers.anonymizer import Anonymizer

num_trees = 500
nltk.download("punkt")
html_cleaner = re.compile("<.*?>")

def multivariate_laplace(dimension: int, epsilon: float) -> np.ndarray:
    """
    Generate a multivariate Laplace noise sample.

    Args:
        dimension (int): The dimension of the noise vector.
        epsilon (float): The privacy parameter.

    Returns:
        np.ndarray: A noise vector sampled from a multivariate Laplace distribution.
    """
    rand_vec = np.random.normal(size=dimension)
    normalized_vec = rand_vec / np.linalg.norm(rand_vec)
    magnitude = np.random.gamma(shape=dimension, scale=1 / epsilon)
    return normalized_vec * magnitude


def madlib(
    review: str, model: Word2Vec, html_cleaner: str, indexer: AnnoyIndexer, epsilon: float
) -> str:
    """
    Apply madlib mechansim to a single text review.

    Args:
        review (str): The input text review.
        model (Word2Vec): The word embedding model.
        html_cleaner (str): Regular expression pattern for cleaning HTML tags.
        indexer (AnnoyIndex): Annoy index for efficient nearest neighbor search.
        epsilon (float): The privacy parameter.

    Returns:
        str: The differentially private version of the input review.
    """
    dimension = model.vectors.shape[-1]
    review = word_tokenize(re.sub(html_cleaner, "", review.lower()))
    priv_words = []
    for word in review:
        if word in model.key_to_index:
            v = model[word]
            per = v + multivariate_laplace(dimension=dimension, epsilon=epsilon)
            priv_word = model.most_similar([per], topn=1, indexer=indexer)[0][0]
            priv_words.append(priv_word)
    priv_review = MosesDetokenizer().detokenize(priv_words, return_str=True)
    return priv_review

class MadlibAnonymizer(Anonymizer):
    def __init__(self, epsilon=10.0, model_name= "glove-wiki-gigaword-50", num_trees=500):
        super().__init__()
        
        self.epsilon = epsilon
        self.model = api.load(model_name)
        self.index = AnnoyIndexer(self.model, num_trees)
        pandarallel.initialize(progress_bar=True)
        
    def anonymize(self, text: str) -> str:
        
        anonymized_text = madlib(
            review=text,
            model=self.model,
            html_cleaner=html_cleaner,
            indexer=self.index,
            epsilon=self.epsilon,
        )
        
        return anonymized_text
