import json
from collections import defaultdict, Counter
import math
import spacy
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "tagger"]) # disable unused components for speed


class NGramModel:
    def __init__(self, n=2):
        self.n = n
        self.ngram_counts = defaultdict(Counter)
        self.context_counts = Counter()
        self.vocab = set()
        self.total_ngrams = 0

    def fit(self, jsonl_path, text_key="text"):
        """
        Fit the n-gram model on a jsonl dataset.
        """
        with open(jsonl_path, "r") as f:
            for line in f:
                obj = json.loads(line)
                text = obj.get(text_key, "")
                tokens = [t.text for t in nlp(text)]
                self.vocab.update(tokens)
                tokens = ["<s>"] * (self.n - 1) + tokens + ["</s>"]
                for i in range(len(tokens) - self.n + 1):
                    context = tuple(tokens[i:i+self.n-1])
                    word = tokens[i+self.n-1]
                    self.ngram_counts[context][word] += 1
                    self.context_counts[context] += 1
                    self.total_ngrams += 1

    def prob(self, text):
        """
        Calculate the log-probability of a given text using the fitted n-gram model (with add-1 smoothing).
        """
        # tokens = text.split()
        tokens = [t.text for t in nlp(text)] # change to spacy tokenizer
        tokens = ["<s>"] * (self.n - 1) + tokens + ["</s>"]
        log_prob = 0.0
        V = len(self.vocab) + 1  # +1 for unseen words
        for i in range(len(tokens) - self.n + 1):
            context = tuple(tokens[i:i+self.n-1])
            word = tokens[i+self.n-1]
            count = self.ngram_counts[context][word]
            total = self.context_counts[context]
            # Add-1 smoothing
            prob = (count + 1) / (total + V)
            log_prob += math.log(prob)
        return log_prob

