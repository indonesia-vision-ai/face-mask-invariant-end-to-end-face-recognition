from functools import partial
from multiprocessing import Pool
from itertools import chain
from pathlib import Path
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


class FRLitePredictor():
    def __init__(self, net, device, num_workers=8, batch_size=64):
        self.net = net
        self.device = device
        self.workers = num_workers
        self.batch_size = batch_size

        self.prediction_list = []

        self.net.to(self.device)
        self.net.eval()


    def evaluate(self, dataset, save_path, save_name):
        with torch.no_grad():
            _, _, _ = self._predict_(dataset)

        save_path = Path(save_path) / (save_name + '.csv')
        save_path.parent.mkdir(exist_ok=True, parents=True)
        save_path = str(save_path)

        preds_df = pd.DataFrame(self.prediction_list,
                                columns=["p1", "p2", "pred", "gt"])
        preds_df.to_csv(save_path, index=False)


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

            self.prediction_list.append([
                str(pair[0]), str(pair[1]), pred, ground_truth
            ])

        return preds, trues, np.mean(inference_times)
