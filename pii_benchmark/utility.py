from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from typing import Tuple


def utility_scores(text: str, reference: str) -> Tuple[int]:
    # Tokenize
    hyp_tokens = nltk.word_tokenize(text)
    ref_tokens = nltk.word_tokenize(reference)

    # BLEU
    smoothie = SmoothingFunction().method1
    bleu = sentence_bleu(
        [ref_tokens],
        hyp_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothie,
    )

    # Rouge
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = scorer.score(reference, text)
    rougeL_f1 = rouge_scores["rougeL"].fmeasure

    return rougeL_f1, bleu
