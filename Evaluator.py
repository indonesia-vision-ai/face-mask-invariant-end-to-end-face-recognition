from multiprocessing import Pool
from functools import partial
from pathlib import Path
from itertools import chain
import time
import json
import pickle

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold

from utils.eval_utils import (roc_auc_threshold, calculate_acc,
                              calculate_cosin_sim,
                              calculate_acc_and_thresh)


class FRLiteEvaluator():
    def __init__(self, net, device, num_workers=8, batch_size=64):
        self.net = net
        self.device = device
        self.workers = num_workers
        self.batch_size = batch_size

        self.net.to(self.device)
        self.net.eval()


    def evaluate(self, dataset, save_path, save_name):
        with torch.no_grad():
            preds, trues, mean_inf_time = self._predict_(dataset)

        fpr, tpr, roc_acc, roc_auc, roc_th = self._evaluate_roc_(preds, trues)

        kfold_acc, kfold_th = self._evaluate_kfold_(preds, trues)

        roc_th = roc_th[0]

        tpr_at_fpr = self._tpr_at_fpr_(fpr, tpr)

        if save_path:
            save_path = Path(save_path) / save_name
            save_path.parent.mkdir(exist_ok=True, parents=True)
            save_path = str(save_path)

            json_dict = {
                'device': str(self.device),
                'inference_time': mean_inf_time,
                'kfold': {
                    'acc': kfold_acc,
                    'th': kfold_th
                },
                'roc': {
                    'auc': roc_auc,
                    'th': roc_th,
                    'acc': roc_acc,
                    'fpr': list(fpr),
                    'tpr': list(tpr),
                    'tpr_at_fpr': tpr_at_fpr
                }
            }

            with open(save_path + '.json', 'w') as jsonfile:
                json.dump(json_dict, jsonfile, indent=4)

            self._create_roc_graph_(fpr, tpr, save_path)


    def _predict_(self, dataset):
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            shuffle=False,
            pin_memory=False
        )

        inference_times = np.array([], dtype=np.float32)
        features = {}
        pbar = tqdm(dataloader,
                    mininterval=2.0, position=0, leave=True,
                    # bar_format='{l_bar}{bar:5}{r_bar}{bar:-5b}'
                    dynamic_ncols=True)
        for imgs, keys in pbar:
            imgs = imgs.to(self.device)

            start = time.time()
            output = self.net(imgs)
            end = time.time() - start
            inference_times = np.append(inference_times, end)

            output = output.cpu().numpy()
            for i, key in enumerate(keys):
                features[str(key)] = output[i]

            pbar.set_description(f"[inf_time {end:.5f}]", refresh=False)
        pbar.close()

        preds = np.array([])
        trues = np.array([])

        pairs_file = list(dataset.pairs_file)
        for pair in pairs_file:
            feat1 = features[str(pair[0])]
            feat2 = features[str(pair[1])]
            ground_truth = int(pair[2])

            pred = calculate_cosin_sim(feat1, feat2)
            preds = np.append(preds, pred)
            trues = np.append(trues, ground_truth)

        return preds, trues, np.mean(inference_times)


    def _evaluate_roc_(self, preds, trues):
        fpr, tpr, auc, threshold = roc_auc_threshold(preds, trues)
        acc = calculate_acc(preds, trues, threshold)
        return fpr, tpr, acc, auc, threshold


    def _evaluate_kfold_(self, preds, trues):
        if len(preds) > 50000:
            return 0., 0. # don't do kfold on gigantic data

        kfold = KFold(n_splits=10, shuffle=False)
        accuracies = []
        best_acc = 0
        threshold = 0

        for train, test in kfold.split(preds):

            _, train_th = calculate_acc_and_thresh(
                preds[train],
                trues[train])

            test_acc = calculate_acc(preds[test],
                                     trues[test],
                                     train_th)

            accuracies.append(test_acc)
            if test_acc > best_acc:
                best_acc = test_acc
                threshold = train_th

        avg_acc = np.mean(accuracies)
        return avg_acc, threshold


    def _create_roc_graph_(self, fpr, tpr, save_path):
        fpr = np.flipud(fpr)
        tpr = np.flipud(tpr)

        plt.style.use('classic')
        plt.figure()
        ax = plt.axes(xscale='log', xlim=[1e-6, 1.0], ylim=[0, 1])
        ax.plot(fpr, tpr, color='blue', lw=1)
        plt.grid(b=True, which='major', axis='x',
                 color='#666666', linestyle='dashed', alpha=0.6)
        plt.grid(b=True, which='minor', axis='x',
                 color='#666666', linestyle='dotted', alpha=0.6)
        plt.grid(b=True, which='major', axis='y',
                 color='#666666', linestyle='solid', alpha=0.7)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')

        if save_path:
            plt.savefig(save_path + '.png')


    def _tpr_at_fpr_(self, fpr, tpr):
        fpr = np.flipud(fpr)
        tpr = np.flipud(tpr)

        tpr_at_fpr = {}
        fpr_range = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        for fpr_iter in np.arange(len(fpr_range)):
            _, min_index = min(
                list(zip(abs(fpr-fpr_range[fpr_iter]),
                         range(len(fpr))))
            )

            tpr_at_fpr[fpr_range[fpr_iter]] = tpr[min_index]

        return tpr_at_fpr
