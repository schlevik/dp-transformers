import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, recall_score, precision_recall_curve, roc_auc_score
import numpy as np
import json
from scipy.stats import beta
from scipy.stats import binomtest
import joblib
import pickle
from matplotlib.lines import Line2D
from attack import NGramReferenceAttack, OnlyNgramAttack, ngram_attack_sampled_datasets_neg_reduced_version
from tqdm import tqdm
import pickle
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def cal_mean_roc_with_ci(y_list, y_hat_list):
    """
    计算多组ROC曲线的均值和95%置信区间
    Args:
        y_list: list of true label lists
        y_hat_list: list of predicted score lists
    Returns:
        [mean_fpr, mean_tpr, lower_tpr, upper_tpr, mean_auc, lower_auc, upper_auc]
    """
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    n = len(y_list)

    for y, y_hat in zip(y_list, y_hat_list):
        fpr, tpr, _ = roc_curve(y, y_hat)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc(fpr, tpr))

    tprs = np.array(tprs)
    aucs = np.array(aucs)
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    # 95% CI
    ci_factor = 1.96 / np.sqrt(n)
    lower_tpr = mean_tpr - ci_factor * std_tpr
    upper_tpr = mean_tpr + ci_factor * std_tpr
    lower_auc = mean_auc - ci_factor * std_auc
    upper_auc = mean_auc + ci_factor * std_auc

    # 限制上下界在[0,1]
    lower_tpr = np.clip(lower_tpr, 0, 1)
    upper_tpr = np.clip(upper_tpr, 0, 1)
    lower_auc = max(lower_auc, 0)
    upper_auc = min(upper_auc, 1)

    resultlist = [mean_fpr, mean_tpr, lower_tpr, upper_tpr, mean_auc, lower_auc, upper_auc]
    return resultlist


def cal_mean_roc(y_list, y_hat_list):
    """
    Calculate ROC AUC for a single y / y_hat pair (no CI needed).
    Args:
        y_list:     list with one array of true labels
        y_hat_list: list with one array of predicted scores
    Returns:
        [mean_fpr, mean_tpr, mean_auc]
    """
    mean_fpr = np.linspace(0, 1, 100)
    fpr, tpr, _ = roc_curve(y_list[0], y_hat_list[0])
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    mean_auc = auc(fpr, tpr)
    return [mean_fpr, interp_tpr, mean_auc]

def find_precision_at_recall(y_scores_list, y_true_list, target_recall=0.8):
    """
    Calculate mean and std of the best precision where recall >= target_recall over multiple runs,
    and also record the corresponding recall value.
    Args:
        y_scores_list: list of arrays/lists of predicted scores
        y_true_list: list of arrays/lists of ground truth labels
        target_recall: recall value at which to report precision
    Returns:
        mean_precision, std_precision, mean_recall, std_recall
    """
    precisions_at_recall = []
    recalls_at_recall = []

    for y_scores, y_true in zip(y_scores_list, y_true_list):
        y_scores = np.array(y_scores)
        y_true = np.array(y_true)
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        valid = recall >= target_recall
        if np.any(valid):
            idx = np.argmax(precision[valid])  # best precision among valid
            valid_indices = np.where(valid)[0]
            best_idx = valid_indices[idx]
            precisions_at_recall.append(precision[best_idx])
            recalls_at_recall.append(recall[best_idx])
        else:
            precisions_at_recall.append(np.nan)
            recalls_at_recall.append(np.nan)

    return (
        np.nanmean(precisions_at_recall),
        np.nanstd(precisions_at_recall),
        np.nanmean(recalls_at_recall),
        np.nanstd(recalls_at_recall)
    )

# ...existing code...

def mia_collect_results_neg_reduced_version(datasetname, baselinename, attackname):
    label_path           = f"/mnt/nvme1/yidan/MIA/data/cls/{datasetname}/D_sample/sampled_private_datasets/sampling_labels.json"
    pos_target_jsonl_dir = f"/mnt/nvme1/yidan/MIA/data/cls/{datasetname}/D_synth/{baselinename}/finalresults/"
    neg_target_jsonl_dir = f"/mnt/nvme1/yidan/MIA/data/cls/{datasetname}/D_synth/{baselinename}/sampled_private_reduced_datasets/"

    paralist   = ["e0", "e4", "e2", "e1", "e0.5"]
    d_idx_list = list(range(100))

    print(f"Dataset: {datasetname}, Baseline: {baselinename}, Attack: {attackname}")

    with open(label_path, "r") as f:
        label_records = json.load(f)

    results_dic = {}

    if attackname == "ngram_reference":
        ori_reference_dir = f"/mnt/nvme1/yidan/MIA/data/cls/{datasetname}/D_sample/sampled_4ref_datasets_outliers/"
        ori_reference_jsonl_paths = [f"{ori_reference_dir}dataset_{k}.jsonl" for k in range(4)]
        ngram = 2
        results_dic[ngram] = {}
        attackmodel = NGramReferenceAttack(ori_reference_jsonl_paths, ngram_n=ngram, text_key="text")

    elif attackname == "ngram_only":
        ngram = 2
        results_dic[ngram] = {}
        attackmodel = OnlyNgramAttack(ngram_n=ngram, text_key="text")

    os.makedirs(f"./results_v2/ngram_reduced/{attackname}/", exist_ok=True)
    for para in paralist:
        out_path = f"./results_v2/ngram_reduced/{attackname}/{datasetname}_{baselinename}_{para}_rerun.pkl"
        if os.path.exists(out_path):
            print(f"  Skip processing existing result for {para} → {out_path}")
            results_dic[ngram][para] = pickle.load(open(out_path, "rb"))
            print(f"  Loaded existing result for {para} → {out_path}")
            print(f"  ngram={ngram} {para} precision@recall0.01: {results_dic[ngram][para][0][4]:.4f}")
            print(f"  ngram={ngram} {para} ROC AUC: {results_dic[ngram][para][0][2]:.4f}")
            continue
        print(f"  epsilon/noise: {para}")

        # Single run — no repeat loop needed (pos/neg mapping is deterministic)
        y, y_hat = ngram_attack_sampled_datasets_neg_reduced_version(
            d_idx_list,
            para,
            pos_target_jsonl_dir,
            neg_target_jsonl_dir,
            label_records,
            attackmodel,
        )

        if len(y) == 0 or len(y_hat) == 0:
            print(f"  [WARN] No samples for {para}, skipping.")
            continue
        assert len(y) == len(y_hat), f"Length mismatch: {len(y)} vs {len(y_hat)}"

        # Wrap in list for compatibility with cal_mean_roc / find_precision_at_recall
        y_list_     = [np.array(y)]
        y_hat_list_ = [np.array(y_hat)]

        mean_p, std_p, mean_r, std_r = find_precision_at_recall(
            y_hat_list_, y_list_, target_recall=0.01)
        print(f"ngram={ngram} {para} precision@recall0.01: {mean_p:.4f}")

        resultlist = cal_mean_roc(y_list_, y_hat_list_)
        print(f"  ROC AUC: {resultlist[2]:.4f}")
        results_dic[ngram][para] = [resultlist, [y_list_, y_hat_list_]]
    
        pickle.dump(results_dic, open(out_path, "wb"), protocol=5)
        print(f"Saved → {out_path}")


def load_results_dic_from_dir(result_dir, datasetname, baselinename, ngram=2, paralist=None):
    if paralist is None:
        paralist = ["e0", "e4", "e2", "e1", "e0.5"]
    results_dic = {ngram: {}}
    for para in paralist:
        file_path = os.path.join(result_dir, f"{datasetname}_{baselinename}_{para}_rerun.pkl")
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                # Each file contains a results_dic, but we only want the [ngram][para] entry
                file_dic = pickle.load(f)
                if ngram in file_dic and para in file_dic[ngram]:
                    results_dic[ngram][para] = file_dic[ngram][para]
                else:
                    print(f"Warning: {file_path} does not contain expected ngram/para keys.")
        else:
            print(f"File not found: {file_path}")
    return results_dic

def plot_multiple_roc_curves_with_boundaries(results_dic, boundaries, paralist, ngram=2, title="ROC Curves for Different Epsilons", save_path=None):
    para_dic = {"e0": "ε=inf", "e4": "ε=4", "e2": "ε=2", "e1": "ε=1", "e0.5": "ε=0.5"}
    bound_keys_dic = {"e0": 0, "e4": 4, "e2": 2, "e1": 1, "e0.5": 0.5}
    plt.figure(figsize=(7, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, len(paralist)))
    for idx, para in enumerate(paralist):
        if para not in results_dic[ngram]:
            continue
        result = results_dic[ngram][para]
        mean_fpr, mean_tpr, mean_auc = result[0][:3]
        plt.plot(mean_fpr, mean_tpr, color=colors[idx], lw=2, label=f"{para_dic.get(para, para)} (AUC={mean_auc:.4f})")
        # Plot theoretical boundary if available
        if bound_keys_dic.get(para) in boundaries:
            bound_fpr, bound_tpr = boundaries[bound_keys_dic.get(para)]
            plt.plot(bound_fpr, bound_tpr, color=colors[idx], lw=2, linestyle=':', label=f"{para_dic.get(para, para)} Theoretical")
    plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()




# attackname = "ngram_reference"
# for datasetname in ["Daniel-ML"]:
#     for baselinename in ["dp-transformers"]:
#         out_path = f"./results_v2/ngram_reduced/{datasetname}_{baselinename}_rerun.pkl"
#         if not os.path.exists(out_path):
#             mia_collect_results_neg_reduced_version(datasetname, baselinename, attackname)
#         else:
#             print(f"Skip existing result for {datasetname} {baselinename}")


# Example usage:
fig_save_dir = "/home/yidan/Projects/MIA/figs/ngram_reduced/"
os.makedirs(fig_save_dir, exist_ok=True)
attackname = "ngram_reference"
result_dir = f"/home/yidan/Projects/MIA/results_v2/ngram_reduced/{attackname}/"
datasetname = "Daniel-ML"
baselinename = "dp-transformers"
paralist = ["e0", "e4", "e2", "e1", "e0.5"]
ngram = 2

# Load theoretical boundaries
with open("/home/yidan/Projects/MIA/results/eps_curves.pkl", "rb") as f:
    boundaries = pickle.load(f)  # should be a dict: {epsilon: (fpr_array, tpr_array)}

results_dic_s2 = load_results_dic_from_dir(result_dir, datasetname, baselinename, ngram, paralist)



# Usage
plot_multiple_roc_curves_with_boundaries(
    results_dic_s2, boundaries, paralist, ngram=ngram,
    title="ROC Curves for Different Epsilons (with Theoretical Boundaries)",
    save_path=f"{fig_save_dir}{datasetname}_{baselinename}_{attackname}_roc_curves_with_boundaries.png"
)



# # Assuming results_dic is already loaded as in your script
# print("\n=== AUC Values (with potential flipping) strategy 2 ===")
# for para in ["e0", "e4", "e2", "e1", "e0.5"]:
#     if para not in results_dic_s2[ngram]:
#         print(f"No results for {para}")
#         continue
#     result = results_dic_s2[ngram][para]
#     y_list, y_hat_list = result[1]
#     y = np.array(y_list[0])
#     y_hat = np.array(y_hat_list[0])
#     # Flip the attack score
#     flipped_score = -y_hat
#     auc_val = roc_auc_score(y, y_hat)
#     auc_val_flipped = roc_auc_score(y, flipped_score)
#     final_auc = max(auc_val, auc_val_flipped)
#     # print(f"Original AUC for {para}: {auc_val:.4f}")
#     # print(f"Flipped AUC for {para}: {auc_val_flipped:.4f}")
#     print(f"Final AUC for {para}: {final_auc:.4f}")

# print("\n=== AUC Values (with potential flipping) strategy 1 ===")
# results_dic_s1 = pickle.load(open(f"/home/yidan/Projects/MIA/results_v2/ngram_noref/{datasetname}_{baselinename}.pkl", "rb"))
# # Assuming results_dic is already loaded as in your script
# for para in ["e0", "e4", "e2", "e1", "e0.5"]:
#     if para not in results_dic_s1[ngram]:
#         print(f"No results for {para}")
#         continue
#     result = results_dic_s1[ngram][para]
#     y_list, y_hat_list = result[1]
#     y = np.array(y_list[0])
#     y_hat = np.array(y_hat_list[0])
#     # Flip the attack score
#     flipped_score = -y_hat
#     auc_val = roc_auc_score(y, y_hat)
#     auc_val_flipped = roc_auc_score(y, flipped_score)
#     final_auc = max(auc_val, auc_val_flipped)
#     # print(f"Original AUC for {para}: {auc_val:.4f}")
#     # print(f"Flipped AUC for {para}: {auc_val_flipped:.4f}")
#     print(f"Final AUC for {para}: {final_auc:.4f}")

