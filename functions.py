#!/usr/bin/env python
# coding: utf-8

# In[110]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    plot_confusion_matrix,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from typing import List
import re
from scipy.stats import ttest_ind, median_test
import statsmodels.stats.api as sms
import statsmodels.api as sm


# In[111]:


def convert_col_name(col_name: str) -> str:
    if col_name.islower():
        return col_name
    list_of_words = re.split("(?=[A-Z])", col_name)
    words_list = [x.lower() for x in list_of_words]
    col = "_".join(words_list)
    return col


# In[112]:


def set_labels(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:

    title = title.replace("_", " ").capitalize()
    xlabel = xlabel.replace("_", " ").capitalize()
    ylabel = ylabel.replace("_", " ").capitalize()

    ax.set_title(title, pad=20, fontsize=14, fontweight="semibold")
    ax.set_xlabel(
        xlabel,
        fontsize=12,
        labelpad=12,
    )
    ax.set_ylabel(
        ylabel,
        fontsize=12,
    )


# In[113]:


def set_bar_pct(ax: plt.Axes, fontsize: str) -> None:
    for p in ax.patches:
        if np.isnan(p.get_height()):
            continue
        _x = p.get_x() + p.get_width() / 2
        _y = p.get_y() + p.get_height()
        value = f"{int(p.get_height())}%"
        ax.text(
            _x,
            _y,
            value,
            verticalalignment="bottom",
            ha="center",
            fontsize=fontsize,
            fontweight="normal",
        )


# In[114]:


def display_step_hists(
    sample_1: pd.DataFrame,
    sample_2: pd.DataFrame,
    list_of_cols: list,
    title: str,
    label_1: str,
    label_2: str,
) -> None:

    fig = plt.figure(figsize=[16, 6])

    fig.suptitle(
        title,
        fontsize=16,
        fontweight="semibold",
        y=1.02,
    )

    for i, column in enumerate(list_of_cols, 1):
        ax = plt.subplot(1, 2, i)
        sns.histplot(
            sample_1[column],
            fill=True,
            legend=True,
            label=label_1,
            element="step",
            binwidth=1,
            alpha=0.7,
            edgecolor=[],
            color="#81bfcf",
        )
        sns.histplot(
            sample_2[column],
            fill=True,
            legend=True,
            label=label_2,
            element="step",
            binwidth=1,
            alpha=0.4,
            edgecolor=[],
            
            color="#444282",
        )
        set_labels(ax, "", column, "density")
        plt.legend()

    plt.tight_layout(pad=1, h_pad=3)


# In[115]:


def compare_means_z(
    sample_1: pd.DataFrame, sample_2: pd.DataFrame, feature_list: list
) -> pd.DataFrame:

    rows = []

    for feature in feature_list:
        cm = sms.CompareMeans(
            sms.DescrStatsW(sample_1[feature]), sms.DescrStatsW(sample_2[feature])
        )
        z_stat, p_val = cm.ztest_ind(usevar="unequal")
        lb, ub = cm.tconfint_diff(usevar="unequal")
        rows.append([feature, p_val, z_stat, lb, ub])

    difference_in_means = pd.DataFrame(
        rows, columns=["feature", "p-value", "z-statistic", "CI lower", "CI upper"]
    )
    difference_in_means = difference_in_means.set_index("feature")

    return difference_in_means


# In[116]:


def display_dists(
    sample_1: pd.DataFrame,
    sample_2: pd.DataFrame,
    list_of_cols: list,
    title: str,
    label_1: str,
    label_2: str,
) -> None:

    fig = plt.figure(figsize=[16, 16])

    fig.suptitle(
        title,
        fontsize=16,
        fontweight="semibold",
        y=1.02,
    )

    for i, column in enumerate(list_of_cols, 1):
        ax = plt.subplot(4, 3, i)
        sns.kdeplot(
            sample_1[column], fill=True, legend=True, label=label_1, color="#444282"
        )
        sns.kdeplot(
            sample_2[column],
            fill=True,
            color="#c13639",
            legend=True,
            label=label_2,
        )

        set_labels(ax, "", column, "density")
        plt.legend()

    plt.tight_layout(pad=1, h_pad=3)


# In[117]:


def compare_means_t(
    sample_1: pd.DataFrame, sample_2: pd.DataFrame, feature_list: list
) -> pd.DataFrame:

    rows = []

    for feature in feature_list:
        stat, p = ttest_ind(sample_1[feature], sample_2[feature], equal_var=False)
        cm = sms.CompareMeans(
            sms.DescrStatsW(sample_1[feature]), sms.DescrStatsW(sample_2[feature])
        )
        lb, ub = cm.tconfint_diff(usevar="unequal")
        rows.append([feature, p, stat, lb, ub])

    difference_in_means = pd.DataFrame(
        rows, columns=["feature", "p-value", "t-statistic", "CI lower", "CI upper"]
    )
    difference_in_means = difference_in_means.set_index("feature")

    return difference_in_means


# In[118]:


def compare_medians(
    sample_1: pd.DataFrame, sample_2: pd.DataFrame, feature_list: list
) -> pd.DataFrame:

    rows = []

    for feature in feature_list:
        stat, p_val = median_test(sample_1[feature], sample_2[feature])[:2]
        cm = sms.CompareMeans(
            sms.DescrStatsW(sample_1[feature]), sms.DescrStatsW(sample_2[feature])
        )
        lb, ub = cm.tconfint_diff(usevar="unequal")
        rows.append([feature, p_val, stat, lb, ub])

    difference_in_medians = pd.DataFrame(
        rows, columns=["feature", "p-value", "z-statistic", "CI lower", "CI upper"]
    )
    difference_in_medians = difference_in_medians.set_index("feature")

    return difference_in_medians


# In[119]:


def evaluate_pvalue(val: int) -> str:

    if val < 0.05:
        return "Stat. significant difference"
    return "Not enough evidence"


# In[120]:


def replace_corr_labels(label: str) -> str:
    if label.startswith("avg"):
        return label[4:].replace("_", " ").capitalize()
    else:
        return label.replace("_", " ").capitalize()


# In[121]:


def get_wins_percentage(
    df: pd.DataFrame, column_1: str, column_2: str, column_3: str
) -> pd.DataFrame:

    ht_num_of_matches = df.groupby(column_1).agg(
        ht_num_of_matches=("match_api_id", "count")
    )

    at_num_of_matches = df.groupby(column_2).agg(
        at_num_of_matches=("match_api_id", "count")
    )

    num_of_matches = ht_num_of_matches.reset_index().merge(
        at_num_of_matches.reset_index(), left_on=column_1, right_on=column_2
    )

    num_of_matches["num_of_matches"] = (
        num_of_matches["ht_num_of_matches"] + num_of_matches["at_num_of_matches"]
    )

    wins_by_team = (
        df.groupby(column_3).agg(count_of_wins=(column_3, "count")).reset_index()
    )

    wins_by_team = wins_by_team.merge(
        num_of_matches[[column_1, "num_of_matches"]],
        left_on=column_3,
        right_on=column_1,
    ).drop(columns=column_1)

    wins_by_team = wins_by_team.rename(columns={column_3: "team"})

    wins_by_team["wins_percentage"] = round(
        ((wins_by_team["count_of_wins"] / wins_by_team["num_of_matches"]) * 100), 2
    )

    return wins_by_team


# In[122]:


def display_hists(
    sample_1: pd.DataFrame,
    sample_2: pd.DataFrame,
    list_of_cols: list,
    title: str,
    label_1: str,
    label_2: str,
    n_locator: int = 6,
    bins: int = 6,
    color_1: str = "#b93540",
    color_2: str = "#4798ce",
) -> None:

    fig = plt.figure(figsize=[15, 15])

    fig.suptitle(
        title,
        fontsize=16,
        fontweight="semibold",
        y=1.02,
    )

    for i, column in enumerate(list_of_cols, 1):
        ax = plt.subplot(4, 3, i)
        sns.histplot(
            sample_1[column],
            bins=bins,
            color=color_1,
            legend=True,
            label=label_1,
            stat="density",
            edgecolor=[],
            alpha=0.5,
        )
        sns.histplot(
            sample_2[column],
            bins=bins,
            color=color_2,
            legend=True,
            label=label_2,
            stat="density",
            edgecolor=[],
            alpha=0.5,
        )

        set_labels(ax, "", column, "density")
        plt.legend()
        ax.xaxis.set_major_locator(plt.MaxNLocator(n_locator))

    plt.tight_layout(pad=1, h_pad=3)


# In[123]:


def set_labels_cm(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:

    title = title.replace("_", " ").capitalize()
    xlabel = xlabel.replace("_", " ").capitalize()
    ylabel = ylabel.replace("_", " ").capitalize()

    ax.set_title(title, pad=10, fontsize=6, fontweight="semibold")
    ax.set_xlabel(xlabel, fontsize=7, labelpad=12)
    ax.set_ylabel(
        ylabel,
        fontsize=7,
    )


# In[124]:


def create_confusion(
    test: pd.Series, pred: list, ticks: List[str], figsize=(2.5, 2.5)
) -> None:

    fig, ax = plt.subplots(figsize=(figsize))

    cm = confusion_matrix(test, pred, normalize="true")
    #  The normalize parameter specifies that the denominator should be 'true' (actuals)

    sns.heatmap(
        cm,
        cmap="RdBu_r",
        annot=True,
        ax=ax,
        cbar=False,
        square=True,
        annot_kws={"fontsize": 6},
        fmt=".1%",
    )

    ax.set_xticklabels(ticks)
    ax.set_yticklabels(ticks)

    set_labels_cm(
        ax,
        "confusion matrix (normalized by actuals)",
        "predicted",
        "actual",
    )

    plt.yticks(rotation=0)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.show()


# In[125]:


def evaluate_lnr_model(y_test: pd.Series, predicted_values: list) -> None:

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    residuals = y_test - predicted_values
    sns.regplot(
        x=predicted_values, y=residuals, color="#b93540", ax=ax[0], truncate=False
    )
    set_labels(ax[0], "residuals against predicted values", "predicted", "residuals")

    sns.histplot(residuals, kde=True, ax=ax[1], bins=20)
    set_labels(ax[1], "residuals distribution", "residuals", "frequency")

    plt.tight_layout()
    plt.show()

