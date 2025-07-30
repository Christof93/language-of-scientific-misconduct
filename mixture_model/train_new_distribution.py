import sys
import time
import json
import math
from collections import Counter

from pathlib import Path

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from data_loader import create_data_split, sub_sample_to_smaller_size
from evaluate_parameter_search import fn_to_setting
from estimation import estimate_text_distribution
from MLE import MLE
from tokenizer import tokenize

SEED = 42
DEFAULT_CONFIG = {
    "gt_alpha": [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
    "n_bootstrap": 1000,
    "validation_cutoff": 0.5,
    "n_workers": 2,
    "aggregation_level":"sentence",
    "sections":["Abstract", "Introduction", "Related Work", "Methods", "Result&Discussion", "Conclusion"],
    "domains": "all",
    "fields":"all",
    "reasons":"all",
    "pos_tags":"all",
    "recalculate_estimate": True,
    "save_output": True,
}

# SPDX-FileCopyrightText: 2022 Idiap Research Institute
#
# SPDX-License-Identifier: MIT

""" Computes the convex hull of Precision-Recall/ROC points returned by sklearn. """


def _compute_stepwise_convex_hull_precision_recall(recall, precision):
    """ Computes the step-wise convex hull for precision-recall points. """
    # for same recall, keep maximum precision
    recall_points = {}
    for r, p in sorted(zip(recall, precision)):
        if r not in recall_points or p > recall_points[r]:
            recall_points[r] = p

    # for same precision, keep maximum recall
    points = [(0, 1)]
    cur_r = 0
    for r, p in sorted(recall_points.items(), key=lambda x: (x[1], x[0]), reverse=True):
        if cur_r < r:
            points.append((r, p))
            cur_r = r
    if (1, 0) not in points:
        points.append((1, 0))

    # create curve points
    curve_points = []
    last_r, last_p = 0, 1
    for r, p in points:
        if len(curve_points) > 0 and last_p > p:
            curve_points.append((last_r, p))
        curve_points.append((r, p))
        last_r = r

    return zip(*curve_points)


def compute_stepwise_convex_hull(x, y, mode='roc'):
    if mode == 'roc':
        x_selected, y_selected = _compute_stepwise_convex_hull_precision_recall([-p + 1 for p in x], y)
        return [-x + 1 for x in x_selected], y_selected
    else:
        return _compute_stepwise_convex_hull_precision_recall(x, y)

def subsample_from_domain(data1, data2):
    sub_sampled_data2 = []
    for domain, n in Counter(data1["Domain"]).items():
        if domain is not None:
            sub_sampled_data2.append(
                data2[data2["Domain"] == domain]
                    .sample(n, random_state=SEED)
            )
    return pd.concat(sub_sampled_data2)

def make_experiment(name, data1, data2, config):
    print(json.dumps(config, indent=3))
    
    if config["reasons"] != "all":      
        data1 = data1[data1["Reason"].str.contains("|".join(config["reasons"]))]
        # data2 = data2.sample(len(data1), random_state=SEED)
        data2 = subsample_from_domain(data1, data2)

    if config["fields"] != "all":
        data1 = data1[data1["Field"].isin(config["fields"])]
        data2 = data2[data2["Field"].isin(config["fields"])]

    if config["domains"] != "all":
        data1 = data1[data1["Domain"].isin(config["domains"])]
        data2 = data2[data2["Domain"].isin(config["domains"])]

    print(f"documents dataset 1: {len(data1)}")
    print(f"documents dataset 2: {len(data2)}")
    start = time.time()

    split1 = create_data_split(
        data1,
        val_cutoff=config["validation_cutoff"],
        text_sources=config["sections"],
        aggregation=config["aggregation_level"],
        include_pos=config["pos_tags"]
    )

    split2 = create_data_split(
        data2,
        val_cutoff=config["validation_cutoff"],
        text_sources=config["sections"],
        aggregation=config["aggregation_level"],
        include_pos=config["pos_tags"],
    )

    train_df1 = pd.DataFrame(pd.Series(split1["train"]), columns=["sentences"])
    train_df2 = pd.DataFrame(pd.Series(split2["train"]), columns=["sentences"])
    train_df1, train_df2 = sub_sample_to_smaller_size(train_df1, train_df2)
    
    val_df1 = pd.DataFrame(pd.Series(split1["validation"]), columns=["inference_sentence"])
    val_df2 = pd.DataFrame(pd.Series(split2["validation"]), columns=["inference_sentence"])
    val_df1, val_df2 = sub_sample_to_smaller_size(val_df1, val_df2)

    print(f"training sentences dataset 1: {len(train_df1)}")
    print(f"validation sentences dataset 1: {len(val_df1)}")
    print()
    print(f"training sentences dataset 2: {len(train_df2)}")
    print(f"validation sentences dataset 2: {len(val_df2)}")
    print()
    if len(train_df1)==0 or len(train_df2)==0 or len(val_df1)==0 or len(val_df2)==0:
        return
    checkpoint1 = time.time()
    print(f"elapsed: {round(checkpoint1 - start,3)}s")
    print("estimating/loading distribution...")
    
    save_path = f"mixture_model/distribution/{name}.parquet"
    if not config["recalculate_estimate"] and Path(save_path).exists():
        pass
    else:
        mixture_distribution = estimate_text_distribution(train_df1, train_df2, save_path)

    mixture_distribution = pd.read_parquet(save_path)
    model = MLE(mixture_distribution, workers=config["n_workers"])
    
    checkpoint2 = time.time()
    print()
    print(f"elapsed: {round(checkpoint2 - checkpoint1, 3)}s")
    print("inference on validation data...")
    return infer_through_range(model, val_df1, val_df2, config)    

def infer_through_range(model, val_df1, val_df2, config):
    results = {}
    for i in config["gt_alpha"]:
        val_size1 = math.floor(len(val_df1) * (1-i))
        val_size2 = math.floor(len(val_df2) * i)
        split_val_df1 = val_df1.sample(val_size1, random_state=SEED)
        split_val_df2 = val_df2.sample(val_size2, random_state=SEED)
        results[str(i)] = model.inference(pd.concat([split_val_df1, split_val_df2]), n_bootstrap=config["n_bootstrap"])
    return results

def generate_all_parameter_configs(base_config=DEFAULT_CONFIG):
    all_configs = []
    section_configs = [
        # intro
        ['Abstract'],
        ['Abstract', 'Introduction'], 
        ['Abstract', 'Introduction', 'Related Work'], 
        ['Abstract', 'Introduction', 'Related Work', 'Methods'],
        ['Introduction'],
        # middle
        ['Methods'],
        ['Related Work', 'Methods'],
        ['Related Work', 'Methods', 'Result&Discussion'],
        ['Introduction', 'Related Work'],
        ['Methods', 'Result&Discussion'],
        #end
        ['Result&Discussion'],
        ['Conclusion'],
        ['Methods', 'Result&Discussion', 'Conclusion'],
        ['Result&Discussion', 'Conclusion'],
        # beginning and end
        ['Abstract', 'Conclusion'],
        ['Introduction', 'Conclusion'],
        ['Abstract', 'Introduction', 'Conclusion'],
        #all
        ['Abstract', 'Introduction', 'Related Work', 'Methods', 'Result&Discussion', 'Conclusion'],
    ]
    pos_tag_settings = [
        "all", 
        ["ADJ"],
        ["ADV"],
        ["NOUN"],
        ["VERB"],
        ["ADJ", "ADV"],
        ["ADJ", "NOUN"],
        ["NOUN", "VERB"],
        ["ADJ", "VERB"],
        ["ADV", "VERB"],
        ["ADJ", "ADV", "VERB"],
        ["ADJ", "ADV", "NOUN"],
        ["ADJ", "NOUN", "VERB"],
        ["ADJ", "ADV", "NOUN", "VERB"],
    ]

    for section_setting in section_configs:
        for pos_tag_setting in pos_tag_settings:
            c = base_config.copy()
            c["pos_tags"] = pos_tag_setting
            c["sections"] = section_setting
            all_configs.append(c)
    
    return all_configs

def generate_all_filter_configs(base_config=DEFAULT_CONFIG):
    all_configs = []
    reason_settings = [
        "all",
        ["Paper Mill"],
        ["Fake Peer Review", "Concerns/Issues with Peer Review"],
        ["Falsification/Fabrication of Data"],
        ["Randomly Generated Content"],
    ]
    field_settings = [
        "all",
        ["Medicine"],
        ["Biochemistry, Genetics and Molecular Biology"],
        ["Computer Science"],
        ["Engineering"],
        ["Environmental Science"],
        ["Social Sciences"],
        ["Materials Science"],
        ["Agricultural and Biological Sciences"],
        ["Business, Management and Accounting"],
        ["Neuroscience"],
        ["Chemistry"],
        ["Psychology"],
        ["Immunology and Microbiology"],
        ["Earth and Planetary Sciences"],
        ["Health Professions"],
        ["Physics and Astronomy"],
        ["Economics, Econometrics and Finance"],
        ["Decision Sciences"],
        ["Mathematics"],
        ["Arts and Humanities"],
        ["Energy"],
        ["Pharmacology, Toxicology and Pharmaceutics"],
        ["Nursing"],
        ["Dentistry"],
        ["Chemical Engineering"],
        ["Veterinary"],
    ]
    domain_settings = [
        "all",
        ["Physical Sciences"],
        ["Health Sciences"],
        ["Life Sciences"],
        ["Social Sciences"],
    ]

    for field_setting in field_settings:
        c = base_config.copy()
        c["fields"] = field_setting
        all_configs.append(c)

    for domain_setting in domain_settings:
        for reason_setting in reason_settings:
            c = base_config.copy()
            c["reasons"] = reason_setting
            c["domains"] = domain_setting
            all_configs.append(c)

    return all_configs

def find_best_config(ret_df, ref_df, configs, results_dir):
    best_error = 1.0
    best_config = None
    for i, config in enumerate(configs):
        try:
            error,_ = run_experiment_pipeline(ret_df, ref_df, config, results_dir)
        except Exception as e:
            print(f"experiment {i} failed due to exception!")
            print(e)
            error = 1.0
        if error<best_error:
            best_config = config
    return best_config

def config_to_filename(config):
    return (
        f"vc_{config['validation_cutoff']}_al_{config['aggregation_level']}_"
        f"s_{('all' if len(config['sections'])==6 else '_'.join(config['sections']))}_"
        f"r_{('all' if config['reasons']=='all' else '_'.join(config['reasons']))}_"
        f"d_{('all' if config['domains']=='all' else '_'.join(config['domains']))}_"
        f"f_{('all' if config['fields']=='all' else '_'.join(config['fields']))}_"
        f"p_{('all' if config['pos_tags']=='all' else '_'.join(config['pos_tags']))}"
    ).replace("/", "|").replace(" ", "-")

def run_experiment_pipeline(retraction_df, reference_df, config, results_dir):
    name = config_to_filename(config)
    if config["save_output"]:
        sys.stdout = open(f"{results_dir}/{name}.txt", "w")

    print(name)
    results = make_experiment(name, retraction_df, reference_df, config)
    if results is None:
        return 1, 1
    print()

    errors = []
    for ratio, (estimate, ci) in results.items():
        errors.append(round(abs(float(ratio)-estimate), 3))
        print(f"ground truth alpha {ratio}: inference {estimate} ({ci}) error: {errors[-1]}")
    mean_error = round(np.mean(errors), 3)
    mean_ci = round(np.mean([ci for _, ci in results.values()]), 3)
    results["mean_error"] = mean_error
    results["mean_ci"] = mean_ci
    print(f"mean error: {mean_error}")
    print(f"mean confidence interval: {mean_ci}")
    
    with open(f"{results_dir}/{name}.json", "w") as json_fp:
        json.dump(results, json_fp, indent=2)
    return mean_error, results


def experiment_with_filtered_datasets(ret_df, ref_df, results_dir, parameter_settings):
    config = DEFAULT_CONFIG.copy()
    config["n_workers"] = 24
    config["save_output"] = False
    i=0
    for best_parameter_setting in parameter_settings:
        configs = generate_all_filter_configs(best_parameter_setting)
        for config in configs:
            i+=1
            print(i)
            try:
                run_experiment_pipeline(ret_df, ref_df, config, results_dir)
            except Exception as e:
                print(f"experiment {i} failed due to exception!")
                print(e)

def evaluate_as_classifier(train_df1, train_df2, config):
    print(f"training sentences dataset 1: {len(train_df1)}")
    print()
    print(f"training sentences dataset 2: {len(train_df2)}")
    print()
    if len(train_df1)==0 or len(train_df2)==0 :
        return
    checkpoint1 = time.time()
    print("estimating/loading distribution...")
    
    save_path = f"mixture_model/distribution/classifier_distribution.parquet"
    if not config["recalculate_estimate"] and Path(save_path).exists():
        pass
    else:
        mixture_distribution = estimate_text_distribution(train_df1, train_df2, save_path)

    mixture_distribution = pd.read_parquet(save_path)
    model = MLE(mixture_distribution, workers=config["n_workers"])
    checkpoint2 = time.time()
    print()
    print(f"elapsed: {round(checkpoint2 - checkpoint1, 3)}s")

    return model

def test_as_classifier(model, test_df1, test_df2, threshold):
    tp = 0
    tn = 0
    fp = 0
    fn = 0 
    for doc in test_df1:
        alpha = 1 - model.inference(doc, n_bootstrap=10)[0]
        if alpha > threshold:
            tp += 1
        else:
            fn += 1
    
    for doc in test_df2:
        alpha = 1 - model.inference(doc, n_bootstrap=10)[0]
        
        if alpha > threshold:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def roc_auc_dqf_classifier(model, test_set):
    y_scores = [1 - model.inference(doc, n_bootstrap=10)[0] for doc,_ in test_set]
    y_true = [l for _,l in test_set]
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    print(fpr)
    print(tpr)
    print(thresholds)
    # Compute AUC-ROC
    auc_roc = roc_auc_score(y_true, y_scores)
    return auc_roc

def pr_rec_auc_dqf_classifier(model, test_set):
    # Compute scores for test set
    y_scores = [1 - model.inference(doc, n_bootstrap=10)[0] for doc, _ in test_set]
    y_true = [l for _, l in test_set]

    # Compute Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    recall, precision = compute_stepwise_convex_hull(recall, precision, mode="pr")

    # Compute AUC-PR
    auc_pr = auc(recall, precision)

    # Find the best threshold based on F1-score
    best_threshold = 0
    best_f1 = 0

    for i, threshold in enumerate(thresholds):
        y_pred = [1 if score >= threshold else 0 for score in y_scores]
        f1 = f1_score(y_true, y_pred)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return auc_pr, precision, recall, best_threshold


def main():
    retraction_df = pd.read_parquet("24_08_22_retractions_tokenized.gzip")
    reference_df = pd.read_parquet("24_12_31_reference_articles_most_cited_tokenized.gzip")
    results_dir = "mixture_model/results/run_2/parameter_search"
    config = DEFAULT_CONFIG.copy()
    config["n_workers"] = 24
    config["save_output"] = False
    # Figure out best combinatoion of pos tags and sections
    # configs = generate_all_parameter_configs(config)
    # _ = find_best_config(retraction_df, reference_df, configs, results_dir)
    
    # Test different datasets filtered by fields, domains and reason
    results_dir = "mixture_model/results/run_2/filtered"
    best_parameter_settings = [
        # fn_to_setting("vc_0.5_al_sentence_s_Abstract_Introduction_Conclusion_r_all_d_all_f_all_p_ADV.json", config),
        # fn_to_setting("vc_0.5_al_sentence_s_Abstract_Introduction_r_all_d_all_f_all_p_ADV_VERB.json", config),
        # fn_to_setting("vc_0.5_al_sentence_s_all_r_all_d_all_f_all_p_ADJ.json", config),
        fn_to_setting("vc_0.5_al_sentence_s_Introduction_Conclusion_r_all_d_all_f_all_p_ADV_VERB.json", config),
    ]
    experiment_with_filtered_datasets(retraction_df, reference_df, results_dir, best_parameter_settings)
    
def evaluate_misconduct_classifier(type_of_misconduct):
    train = pd.read_parquet(f"train_splits/train_{type_of_misconduct}_samples.gzip")
    val = pd.read_parquet(f"dev_splits/dev_{type_of_misconduct}_samples.gzip")
    test = pd.read_parquet(f"test_splits/test_{type_of_misconduct}_samples.gzip")

    retracted_train_fn = f"mixture_model/classifier_distributions/{type_of_misconduct}_train_retracted.parquet"
    nonretracted_train_fn = f"mixture_model/classifier_distributions/{type_of_misconduct}_train_nonretracted.parquet"       

    if Path(retracted_train_fn).exists():
        train1 = pd.read_parquet(retracted_train_fn)
    else:
        train1 = pd.DataFrame.from_dict({
            "sentences": [[w for w,_ in s] for text, label in zip(train["text"], train["label"]) if label == 1 for s in tokenize(text)]
        }) 
        train1.to_parquet(retracted_train_fn, index=False)
    
    if Path(nonretracted_train_fn).exists():
        train2 = pd.read_parquet(nonretracted_train_fn)
    else:
        train2 = pd.DataFrame.from_dict({
            "sentences": [[w for w,_ in s] for text, label in zip(train["text"], train["label"]) if label == 0 for s in tokenize(text)]
        }) 
        train2.to_parquet(nonretracted_train_fn, index=False)   
    # Prepare validation dataset
    valset = [
        (
            pd.DataFrame.from_dict({
                "inference_sentence": [[w for w, _ in s] for s in tokenize(text)]
            }),
            label
        )
        for text, label in zip(val["text"], val["label"])
    ]

    testset = [
        (
            pd.DataFrame.from_dict({
                "inference_sentence": [[w for w,_ in s] for s in tokenize(text)]
            }),
            label
        )
        for text, label in zip(test["text"], test["label"]) 
    ]

    config = DEFAULT_CONFIG.copy()
    config["n_workers"] = 32
    config["save_output"] = False
    
    model = evaluate_as_classifier(train1, train2, config)
    print(f"evaluating on {len(val)} validation sentences")
    pr_rec_auc, precision, recall, threshold  = pr_rec_auc_dqf_classifier(model, valset)
    print(f"Precision-Recall AUC (Validation): {pr_rec_auc:.4f}")
    print(f"\nTest Set Evaluation at Best Threshold ({threshold:.4f}):")

    print(f"evaluating on {len(testset)} test sentences")
    p, r, f1 = test_as_classifier(model, [t for t, l in testset if l == 1], [t for t, l in testset if l == 0], threshold)
    print(f"Precision: {p:.4f}")
    print(f"Recall: {r:.4f}")
    print(f"F1 Score: {f1:.4f}")
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=f'Precision Recall Curve (AUC = {pr_rec_auc:.2f})', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guessing', line=dict(color='red', dash='dash')))

    # Customize layout
    fig.update_layout(
        title="Precision Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis=dict(range=[0, 1]),  # Set x-axis from 0 to 1
        yaxis=dict(range=[0, 1]),  # Set y-axis from 0 to 1
        template="plotly_white",
        legend=dict(
            x=1,  # Position at the left
            y=0,  # Position at the bottom
            xanchor="right",
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.5)"  # Semi-transparent background
        ),
        width=900,
        height=900,
        font=dict(size=18)
    )
    # Show plot
    fig.write_image(f"precision_recall_curve_{type_of_misconduct}.png", scale=3)
    return {
        "pr_rec_auc": pr_rec_auc,
        "best_threshold": threshold,
        "precision": p,
        "recall": r,
        "f1_score": f1
    }

def main_classifier():
    for misconduct in [
        "paper_mill", 
        "falsification", 
        "random_content"
    ]:
        print(f"evaluating DQF as {misconduct} detector:")
        evaluate_misconduct_classifier(misconduct)

if __name__=="__main__":
    main_classifier()