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
    "all": "finetune all",
    "classifier": "finetune pred. head",
    "lastbert": "finetune last layer + pred. head",
}
# font = {
#     'weight': 'bold',
# }
# matplotlib.rc('font', **font)


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
    fig, ax = plt.subplots()
    plt.title(name_mapping[metric])
    models = sorted(list(results.keys()))
    for id_plt in models:
        metric_values = results[id_plt]
        values = metric_values[metric]
        lt = line_types[id_plt.split(" - ")[0]]
        ax.plot(range(len(values)), values, label=id_plt, ls=lt)
    ax.legend(bbox_to_anchor=(1.85, 0.5), loc='right')
    plt.xlabel('Epoch')
    plt.savefig(os.path.join(fig_dir, f'{metric}.png'), dpi=300, bbox_inches='tight')


def plot_metric_over_phenotypes(results, is_train, metric, fig_dir, hatch_types):
    """Plots final metric value of all models on different phenotypes.

    Args:
        metric: Can be one of [f1_macro, f1_micro, f1_weighted, accs]."""
    fig = plt.figure(figsize=(30, 15))
    fig.suptitle(f'{"Train" if is_train else "Evaluation"} {name_mapping[metric]}', fontsize=20)
    frame1 = plt.gca()
    frame1.axes.xaxis.set_visible(False)
    frame1.axes.yaxis.set_visible(False)
    plt.box(False)
    models = sorted(list(results.keys()))
    np.random.seed(9)
    colors = np.random.rand(len(models), 3).tolist()
    hatches = [hatch_types[model.split(' - ')[0]] for model in models]
    for i, p in enumerate(data_util.PHENOTYPE_NAMES):
        full_metric_name = f'{"tr" if is_train else "eval"}_{p}_{metric}'
        ax = fig.add_subplot(3, 5, i + 1)
        ax.bar(
            models, [results[m][full_metric_name][-1] for m in models],
            color=colors, alpha=0.5, hatch=hatches)
        ax.title.set_text(p)
        # if i < 10:
        x_axis = ax.axes.get_xaxis()
        x_axis.set_visible(False)
        # else:
        #     plt.xticks(rotation=30, ha='right')
        ax.set_ylim(0.5, 1)
    handles = [plt.Rectangle((0,0),1,1, color=colors[i], alpha=0.5) for i in range(len(colors))]
    plt.legend(handles, models, bbox_to_anchor=(2.1, 0.5), loc='right')
    plt.savefig(os.path.join(fig_dir, f'{"tr" if is_train else "eval"}_{metric}_by_phenotypes.png'), dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    RESULT_DIR = "results"
    FIG_DIR = "figs"
    line_types = {
        "bert-mini": "-",
        "bert-base-cased": "-.",
        "bio+clinicalbert": "--",
    }
    hatch_types = {
        "bert-mini": "-",
        "bert-base-cased": "/",
        "bio+clinicalbert": "\\",
    }
    results = {}
    for f in glob.glob(os.path.join(RESULT_DIR, "*/*.json")):
        fattrs = parse_attrs(f)
        if fattrs["baseline_bert"] == "True":
            id_plt = f"bert-mini - train from scratch (lr={fattrs['lr']})"
        else:
            id_plt = f'{fattrs["bert_name"]} - {name_mapping[fattrs["ft_mode"]]}'
        result_json = json.load(open(f))
        results.update({id_plt: result_json})

    plot_metric_over_epochs(results, "tr_loss", FIG_DIR, line_types)
    plot_metric_over_epochs(results, "eval_loss", FIG_DIR, line_types)
    plot_metric_over_epochs(results, "tr_avg_f1s_weighted", FIG_DIR, line_types)
    plot_metric_over_epochs(results, "eval_avg_f1s_weighted", FIG_DIR, line_types)

    plot_metric_over_phenotypes(results, is_train=False, metric="accs", fig_dir=FIG_DIR, hatch_types=hatch_types)
    plot_metric_over_phenotypes(results, is_train=False, metric="f1_weighted", fig_dir=FIG_DIR, hatch_types=hatch_types)
    plot_metric_over_phenotypes(results, is_train=False, metric="f1_macro", fig_dir=FIG_DIR, hatch_types=hatch_types)
    plot_metric_over_phenotypes(results, is_train=False, metric="f1_micro", fig_dir=FIG_DIR, hatch_types=hatch_types)
