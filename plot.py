import random
from collections import defaultdict
import glob
import json
import os.path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import data_util


name_mapping = {
    "tr_loss": "Train Loss",
    "eval_loss": "Evaluation Loss",
    "tr_avg_f1s_weighted": "Train Average Weighted F1",
    "eval_avg_f1s_weighted": "Evaluation Average Weighted F1",
    "accs": "Accuracy",
    "f1_weighted": "F1 Weighted",
    "f1_macro": "F1 Macro",
    "f1_micro": "F1 Micro",
    "all": "train all",
    "classifier": "train pred. head",
    "lastbert": "train last layer + pred. head",
}


def parse_attrs(fname: str):
    f = fname.split("/")[-1]
    fs = f.split("__")
    attributes = {}
    for a in fs:
        if "=" in a:
            ks = a.split("=")
            attributes.update({ks[0]: ks[1]})
    return attributes


def plot_metric_over_epochs(results, metric, fig_dir, line_types):
    font = {
        'size': 14,
    }
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots()
    plt.title(name_mapping[metric])
    models = sorted(list(results.keys()))
    for id_plt in models:
        metric_values = results[id_plt]
        values = metric_values[metric]
        lt = line_types[id_plt.split(" - ")[0]]
        ax.plot(range(len(values)), values, label=id_plt, ls=lt)
        # print(id_plt, f'{values[-1]:.03f}')
    ax.legend(bbox_to_anchor=(2.2, 0.5), loc='right')
    plt.xlabel('Epoch')
    plt.savefig(os.path.join(fig_dir, f'{metric}.png'), dpi=300, bbox_inches='tight')


def plot_metric_over_phenotypes(results, is_train, metric, fig_dir):
    """Plots final metric value of all models on different phenotypes.

    Args:
        metric: Can be one of [f1_macro, f1_micro, f1_weighted, accs]."""
    font = {
        'size': 15,
    }
    matplotlib.rc('font', **font)
    fig = plt.figure(figsize=(30, 20))
    fig.suptitle(f'{"Train" if is_train else "Evaluation"} {name_mapping[metric]}', fontsize=28)
    frame1 = plt.gca()
    frame1.axes.xaxis.set_visible(False)
    frame1.axes.yaxis.set_visible(False)
    plt.box(False)
    models = sorted(list(results.keys()))
    np.random.seed(9)
    colors = np.random.rand(len(models), 3).tolist()
    print(models)
    hatches = [random.choice(all_hatch_types) for _ in models]
    accs = defaultdict(float)
    for i, p in enumerate(data_util.PHENOTYPE_NAMES):
        full_metric_name = f'{"tr" if is_train else "eval"}_{p}_{metric}'
        ax = fig.add_subplot(3, 5, i + 1)
        ax.bar(
            models, [results[m][full_metric_name][-1] for m in models],
            color=colors, alpha=0.5, hatch=hatches)
        ax.set_title(" ".join(p.split('.')).title(), pad=15)
        x_axis = ax.axes.get_xaxis()
        x_axis.set_visible(False)
        ax.set_ylim(0.5, 1)
        for m in models:
            accs[m] += results[m][full_metric_name][-1]
    for k in accs:
        accs[k] = accs[k] / len(data_util.PHENOTYPE_NAMES)
    for k, v in accs.items():
        print(k, f'{v:.03f}')
    handles = [plt.Rectangle(
        (0,0),1,1, facecolor=colors[i], edgecolor='black', alpha=0.5, hatch=hatches[i]) for i in range(len(colors))]
    plt.legend(handles, models, bbox_to_anchor=(-0.5, -0.8), loc='lower center', ncols=3)
    plt.savefig(os.path.join(fig_dir, f'{"tr" if is_train else "eval"}_{metric}_by_phenotypes.png'), dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    RESULT_DIR = "results"
    FIG_DIR = "figs"
    random.seed(42)
    all_hatch_types = ['+', 'x', '.', 'o', '']
    results = {}
    for f in glob.glob(os.path.join(RESULT_DIR, "*/*.json")):
        fattrs = parse_attrs(f)
        if fattrs["baseline_bert"] == "True":
            id_plt = f"bert-mini - train from scratch (lr={fattrs['lr']})"
        else:
            id_plt = f'{fattrs["bert_name"]} - {name_mapping[fattrs["ft_mode"]]}'
        result_json = json.load(open(f))
        results.update({id_plt: result_json})

    # plot_metric_over_epochs(results, "tr_loss", FIG_DIR, line_types)
    # plot_metric_over_epochs(results, "eval_loss", FIG_DIR, line_types)
    # plot_metric_over_epochs(results, "tr_avg_f1s_weighted", FIG_DIR, line_types)
    # plot_metric_over_epochs(results, "eval_avg_f1s_weighted", FIG_DIR, line_types)

    print("ACCURACIES ==================")
    plot_metric_over_phenotypes(results, is_train=False, metric="accs", fig_dir=FIG_DIR)
    print("F1 WEIGHTED ===================")
    plot_metric_over_phenotypes(results, is_train=False, metric="f1_weighted", fig_dir=FIG_DIR)
    print("F1 MACRO =================")
    plot_metric_over_phenotypes(results, is_train=False, metric="f1_macro", fig_dir=FIG_DIR)
    print("F1 MICRO ==================")
    plot_metric_over_phenotypes(results, is_train=False, metric="f1_micro", fig_dir=FIG_DIR)
