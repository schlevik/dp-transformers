from ngram import NGramModel
import numpy as np
import os
import random

class NGramReferenceAttack:
    def __init__(self, reference_jsonl_paths, ngram_n=2, text_key="text"):
        """
        Fit M ngram models on reference datasets (D').
        """
        self.text_key = text_key
        self.reference_models = []
        for ref_path in reference_jsonl_paths:
            model = NGramModel(n=ngram_n)
            model.fit(ref_path, text_key=text_key)
            self.reference_models.append(model)

    def fit_target(self, target_jsonl_path):
        """
        Fit ngram model on the target dataset (D).
        """
        self.target_model = NGramModel(n=self.reference_models[0].n)
        self.target_model.fit(target_jsonl_path, text_key=self.text_key)

    def predict_logit(self, sample_text):
        """
        For a given sample x, compute:
          - a: probability of x under target model
          - b: average probability of x under reference models
        Return logit = log(a/b)
        """
        ref_probs = [model.prob(sample_text) for model in self.reference_models]
        avg_ref_prob = np.mean(ref_probs)
        target_prob = self.target_model.prob(sample_text)
        logit = target_prob - avg_ref_prob  # log(a/b)
        return logit


class OnlyNgramAttack:
    def __init__(self, ngram_n=2, text_key="text"):
        """
        Initialize ngram model.
        """
        self.ngram_n = ngram_n
        self.text_key = text_key

    def fit_target(self, target_jsonl_path):
        """
        Fit ngram model on the target dataset (D).
        """
        self.target_model = NGramModel(n=self.ngram_n)
        self.target_model.fit(target_jsonl_path, text_key=self.text_key)

    def predict_logit(self, sample_text):
        """
        For a given sample x, compute:
          - a: probability of x under target model
        Return logit = log(a)
        """
        target_prob = self.target_model.prob(sample_text)
        logit = target_prob  # log(a)
        return logit

def ngram_attack_sampled_datasets_neg_reduced_version(d_idx_list,
    epsilon,
    pos_target_jsonl_dir,
    neg_target_jsonl_dir,
    label_records,
    attackmodel,
):

    y = []
    y_hat = []
    posnum = 0
    negnum = 0

    for idx in d_idx_list:
        # Get label info for this dataset
        record = label_records[idx]
        if record["added_special"]:
            print(f"Processing dataset index {idx}/{len(d_idx_list)} with epsilon {epsilon}...")
            # positive sample pair, i.e., the target sample is added to private dataset to generate the synthetic dataset
            pos_target_json_path = pos_target_jsonl_dir + f"dataset_{idx}_{epsilon}_samples.jsonl"
            if os.path.exists(pos_target_json_path):
                posnum += 1
                attackmodel.fit_target(pos_target_json_path)
                special_text = record["special_text"]
                y.append(1)
                logit = attackmodel.predict_logit(special_text)
                y_hat.append(logit)
            # negative sample pair, i.e., the target sample is removed from private dataset to generate the synthetic dataset
            neg_target_json_path = neg_target_jsonl_dir + f"dataset_{idx}_{epsilon}_samples.jsonl"
            if os.path.exists(neg_target_json_path):
                negnum += 1
                attackmodel.fit_target(neg_target_json_path)
                special_text = record["special_text"]
                y.append(0)
                logit = attackmodel.predict_logit(special_text)
                y_hat.append(logit)
            print(f"  Processed {idx}: posnum={posnum}, negnum={negnum}")
            
    return y, y_hat