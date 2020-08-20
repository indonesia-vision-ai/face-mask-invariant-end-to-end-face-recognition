from pathlib import Path
import argparse
import json
import glob
import sys

from matplotlib import pyplot as plt
import numpy as np


def roc_graphs(fprs, tprs, names, aucs, savename, minx=0.85):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    plt.figure()  # figsize=(10, 10)

    ax = plt.axes(xscale='log', xlim=[1e-4, 1.0], ylim=[minx-0.05, 1])

    for i, (fpr, tpr, name, auc) in enumerate(zip(fprs, tprs, names, aucs)):
        fpr = np.flipud(fpr)
        tpr = np.flipud(tpr)
        auc *= 100
        ax.plot(fpr, tpr, color=colors[i], lw=2,
                label=f'{name} (AUC: {auc:.2f}%)')

    plt.grid(b=True, which='major', axis='x',
             color='#666666', linestyle='dashed', alpha=0.6)
    plt.grid(b=True, which='minor', axis='x',
             color='#666666', linestyle='dotted', alpha=0.4)
    plt.grid(b=True, which='major', axis='y',
             color='#999999', linestyle='solid', alpha=0.1)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    ax.legend(loc='lower right', fontsize=10,
              fancybox=True).get_frame().set_alpha(0.5)
    plt.savefig(savename + '_roc.pdf', bbox_inches='tight')


def acc_plot(accs, names, savename):
    plt.figure()  # figsize=(8, 3)
    accs = np.array(accs) * 100
    minacc = min(accs)
    maxacc = max(accs)

    ax = plt.axes(xlim=[minacc-0.5, maxacc+0.5])
    bars = ax.barh(names, accs)
    plt.grid(b=True, which='major', axis='x',
             color='#666666', linestyle='dashed', alpha=0.6)
    plt.title('Thresholded Accurracy')
    plt.xlabel('Accuracy (%)\nHigher is better')

    for name, acc in zip(names, accs):
        plt.text(s=f'{acc:.2f}%', x=acc-0.175, y=name, color="r",
                 verticalalignment="center", size=9)

    bars[np.argmax(accs)].set_color('green')

    plt.tight_layout()
    plt.savefig(savename + '_acc.pdf')


def inftime_plot(inftimes, names, savename):
    plt.figure()  # figsize=(8, 3)
    inftimes = np.array(inftimes) * 1000
    mintime = np.min(inftimes)
    maxtime = np.max(inftimes)

    ax = plt.axes(xlim=[mintime-5, maxtime+1])
    bars = ax.barh(names, inftimes)
    plt.grid(b=True, which='major', axis='x',
             color='#666666', linestyle='dashed', alpha=0.6)
    plt.title('Inference time')
    plt.xlabel('Inference time (ms)\nLower is better')

    for name, time in zip(names, inftimes):
        plt.text(s=f'{time:.2f}ms', x=time-1.75, y=name, color="r",
                 verticalalignment="center", size=9)

    bars[np.argmin(inftimes)].set_color('green')

    plt.tight_layout()
    plt.savefig(savename + '_time.pdf')


def tpr_at_fpr_plot(tpr_at_fprs, names, savename):
    plt.figure()  # figsize=(8, 3)
    tpr_at_fprs = np.array(tpr_at_fprs)
    minval = np.min(tpr_at_fprs)
    maxval = np.max(tpr_at_fprs)

    ax = plt.axes(xlim=[minval-0.05, maxval+0.05])
    bars = ax.barh(names, tpr_at_fprs)
    plt.grid(b=True, which='major', axis='x',
             color='#666666', linestyle='dashed', alpha=0.6)
    plt.title('Verification TAR (@FAR=1e-4)')
    plt.xlabel('TAR (@FAR=1e-4)')

    for name, val in zip(names, tpr_at_fprs):
        plt.text(s=f'{val:.4f}', x=val-0.015, y=name, color="r",
                 verticalalignment="center", size=9)

    bars[np.argmax(tpr_at_fprs)].set_color('green')

    plt.tight_layout()
    plt.savefig(savename + '_TAR.pdf')


def main():
    parser = argparse.ArgumentParser(
        description="Compare evaluation results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--src", "-s", type=str, required=True,
                        help="path to dir with evaluation .json results")
    parser.add_argument("--name", "-n", type=str, required=True,
                        help="name for output file")

    args = parser.parse_args()

    fprs, tprs, accs, theshs, aucs, inftimes = [], [], [], [], [], []
    names, tpr_at_fprs = [], []
    min_tpr = 1.

    src_path = Path(args.src)
    jpaths = glob.glob(str(src_path / '*.json'))

    if len(jpaths) > 7:
        print('Cannot compare more than 7!')
        sys.exit()

    for jpath in jpaths:
        with open(jpath, 'r') as jfile:
            jsondata = json.load(jfile)

        name = Path(jpath).stem
        names.append(name)
        fprs.append(jsondata['roc']['fpr'])
        tprs.append(jsondata['roc']['tpr'])

        kfold_acc = jsondata['kfold']['acc']
        roc_acc = jsondata['roc']['acc']

        if kfold_acc > roc_acc:
            accs.append(kfold_acc)
            theshs.append(jsondata['kfold']['th'])
        else:
            accs.append(roc_acc)
            theshs.append(jsondata['roc']['th'])

        aucs.append(jsondata['roc']['auc'])
        inftimes.append(jsondata['inference_time'])
        tpr_at_fprs.append(jsondata['roc']['tpr_at_fpr']['0.0001'])

        min_tpr = min(jsondata['roc']['tpr_at_fpr']['1e-06'], min_tpr)

    Path(args.name).parent.mkdir(parents=True, exist_ok=True)

    roc_graphs(fprs, tprs, names, aucs, args.name, min_tpr)
    acc_plot(accs, names, args.name)
    tpr_at_fpr_plot(tpr_at_fprs, names, args.name)
    inftime_plot(inftimes, names, args.name)


if __name__ == "__main__":
    main()
