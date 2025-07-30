import itertools
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

from mixture_model.data_loader import create_data_split


def log_odds_analysis(
    retraction_path,
    reference_path, 
    field="all",
    domain="all",
    sections=["Introduction"],
    pos=["ADV"],
    ngrams = 1
):
    retraction_df = pd.read_parquet(retraction_path)
    reference_df = pd.read_parquet(reference_path)

    if field != "all":
        retraction_df = retraction_df[retraction_df["Field"] == field]
        reference_df = reference_df[reference_df["Field"] == field]
    if domain != "all":
        retraction_df = retraction_df[retraction_df["Domain"] == domain]
        reference_df = reference_df[reference_df["Domain"] == domain]

    for section in sections:
        retraction_df = retraction_df[retraction_df[f"{section} POS Tags"].notna()]
        reference_df = reference_df[reference_df[f"{section} POS Tags"].notna()]

    if retraction_df.empty or reference_df.empty:
        print(f"No data available for field: {field}, domain: {domain}. Skipping...")
        return

    def normalize_adverbs(adverbs_df, smoothing_coeff=1):
        total = adverbs_df["Count"].sum()
        if total == 0:
            return adverbs_df
        adverbs_df["Normalized"] = (adverbs_df["Count"] + smoothing_coeff) / (
            total + (smoothing_coeff * len(adverbs_df["Count"]))
        )
        return adverbs_df

    def calculate_log_odds(retracted_adverbs_df, reference_adverbs_df, smoothing_coeff=1):
        merged_adverbs_df = pd.merge(
            retracted_adverbs_df,
            reference_adverbs_df,
            on="Adverb",
            how="outer",
            suffixes=("_retracted", "_reference"),
        )
        n_unique_adverbs = len(
            pd.concat(
                [retracted_adverbs_df["Adverb"], reference_adverbs_df["Adverb"]]
            ).unique()
        )
        zero_occurrence_prob_ref = smoothing_coeff / (
            reference_adverbs_df["Count"].sum() + (smoothing_coeff * n_unique_adverbs)
        )
        zero_occurrence_prob_ret = smoothing_coeff / (
            retracted_adverbs_df["Count"].sum() + (smoothing_coeff * n_unique_adverbs)
        )

        merged_adverbs_df["Normalized_reference"] = merged_adverbs_df[
            "Normalized_reference"
        ].fillna(zero_occurrence_prob_ref)
        merged_adverbs_df["Normalized_retracted"] = merged_adverbs_df[
            "Normalized_retracted"
        ].fillna(zero_occurrence_prob_ret)
        merged_adverbs_df = merged_adverbs_df.fillna(0)

        retracted_odds = merged_adverbs_df["Normalized_retracted"] / (
            1 - merged_adverbs_df["Normalized_retracted"]
        )
        reference_odds = merged_adverbs_df["Normalized_reference"] / (
            1 - merged_adverbs_df["Normalized_reference"]
        )
        merged_adverbs_df["Log-Odds"] = np.log(retracted_odds / reference_odds)

        def chi_squared(row):
            observed = np.array([row["Count_retracted"], row["Count_reference"]])
            expected = observed.sum() / 2
            chi2, p = chi2_contingency([observed, [expected, expected]])[:2]
            return p

        merged_adverbs_df["Chi-Squared P-Value"] = merged_adverbs_df.apply(
            chi_squared, axis=1
        )
        return merged_adverbs_df

    if ngrams>1:
        reference_adverbs_count = Counter(
            itertools.chain(*[
                create_ngrams(s, ngrams) for s in create_data_split(reference_df, 1.0, text_sources=sections, include_pos="all")["validation"]
            ])
        )
        retracted_adverbs_count = Counter(
            itertools.chain(*[
                create_ngrams(s, ngrams) for s in create_data_split(retraction_df, 1.0, text_sources=sections, include_pos="all")["validation"]
            ])
        )
    elif len(pos)==1:
        reference_adverbs_count = Counter(
            itertools.chain(*create_data_split(reference_df, 1.0, text_sources=sections, include_pos=pos)["validation"])
        )
        retracted_adverbs_count = Counter(
            itertools.chain(*create_data_split(retraction_df, 1.0, text_sources=sections, include_pos=pos)["validation"])
        )
    elif len(pos)>1:
        reference_adverbs_count = Counter(
            [tuple(s) for s in create_data_split(reference_df, 1.0, text_sources=sections, include_pos=pos)["validation"]]
        )
        retracted_adverbs_count = Counter(
            [tuple(s) for s in create_data_split(retraction_df, 1.0, text_sources=sections, include_pos=pos)["validation"]]
        )
    retracted_adverbs_df = pd.DataFrame.from_dict(
        retracted_adverbs_count, orient="index", columns=["Count"]
    ).reset_index()
    retracted_adverbs_df.rename(columns={"index": "Adverb"}, inplace=True)

    reference_adverbs_df = pd.DataFrame.from_dict(
        reference_adverbs_count, orient="index", columns=["Count"]
    ).reset_index()
    reference_adverbs_df.rename(columns={"index": "Adverb"}, inplace=True)
    print(retracted_adverbs_df["Count"].sum())
    print(reference_adverbs_df["Count"].sum())
    if retracted_adverbs_df["Count"].sum() == 0 or reference_adverbs_df["Count"].sum() == 0:
        print(f"No adverbs found for field: {field}, domain: {domain}. Skipping...")
        return

    retracted_adverbs_df = normalize_adverbs(retracted_adverbs_df)
    reference_adverbs_df = normalize_adverbs(reference_adverbs_df)
    merged_adverbs_df = calculate_log_odds(retracted_adverbs_df, reference_adverbs_df)

    if merged_adverbs_df is None:
        return

    distinguishing_adverbs = merged_adverbs_df[
        (merged_adverbs_df["Chi-Squared P-Value"] < 0.05)
        & (
            (merged_adverbs_df["Log-Odds"] > 1)
            | (merged_adverbs_df["Log-Odds"] < -1)
        )
    ].sort_values(by="Log-Odds", ascending=False)

    if distinguishing_adverbs.empty:
        print(f"No distinguishing adverbs found for field: {field}, domain: {domain}. Skipping plot.")
        return

    print(f"Found {len(distinguishing_adverbs)} distinguishing adverbs for field: {field}, domain: {domain}")

    return distinguishing_adverbs


def visualize(field, domain, distinguishing_adverbs):
    top_5_retracted = distinguishing_adverbs.sort_values("Log-Odds", ascending=False).head(10)
    plt.figure(figsize=(10, 5))
    plt.barh(top_5_retracted["Adverb"], top_5_retracted["Log-Odds"], color="blue", alpha=0.7)
    plt.xlabel("Log-Odds")
    plt.ylabel("Adverbs")
    plt.title(f"Top Adverbs Indicating Retraction (Field: {field}, Domain: {domain})")
    plt.gca().invert_yaxis()
    plt.show()

    top_5_non_retracted = distinguishing_adverbs.sort_values("Log-Odds").head(10)
    plt.figure(figsize=(10, 5))
    plt.barh(top_5_non_retracted["Adverb"], top_5_non_retracted["Log-Odds"], color="green", alpha=0.7)
    plt.xlabel("Log-Odds")
    plt.ylabel("Adverbs")
    plt.title(f"Top Adverbs Indicating Non-Retraction (Field: {field}, Domain: {domain})")
    plt.gca().invert_yaxis()
    plt.show()

def create_ngrams(sentence, n):
    ngrams = zip(*[sentence[i:] for i in range(n)])
    return ngrams

if __name__=="__main__":
    for ngram in create_ngrams(["The", "great", "brown", "fox", "jumps", "over", "the", "fence"], 3):
        print(ngram)