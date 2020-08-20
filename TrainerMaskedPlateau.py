from pathlib import Path
import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils.eval_utils import (roc_auc_threshold, calculate_acc,
                              calculate_cosin_sim)

from datasets.eval.lfw import LFWMasked


WEIGHT_SAVE_DIR = "weights/"
GRAPHS_SAVE_DIR = "viz/"

class FRTrainerPlateau():

    def __init__(self, param_config, net_config, dataset_config,
                 optim_params_func=None, scheduler_params=None,
                 eval_transform=None):

        self.params = param_config
        self.net_config = net_config
        self.datasets = dataset_config
        self.eval_transform = eval_transform

        self.device = torch.device(self.params['device'])

        print('preparing dataloader . . .')
        self.train_loader = torch.utils.data.DataLoader(
            self.datasets['train_dataset'],
            batch_size=self.params['batch_size'],
            num_workers=self.params['workers'],
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )
        print('dataloader ready.')

        self.net = net_config['net']
        self.head = net_config['head']
        self.criterion = net_config['criterion']

        if optim_params_func is None:
            optim_params = self.net.params + self.head.params
        else:
            optim_params = optim_params_func(self.net, self.head, self.params)

        self.optimizer = torch.optim.SGD(
            optim_params,
            lr=self.params['lr'],
            momentum=self.params['momentum']
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', **scheduler_params)

        if self.params['multi_gpu']:
            print('initializing data parallel . . .')
            self.net = torch.nn.DataParallel(self.net)
            self.head = torch.nn.DataParallel(self.head)


        self.net.to(self.device)
        self.head.to(self.device)

        self.total_iter = 0


    def train(self):
        print('training started.')
        training_losses = np.array([])
        eval_accs = np.array([])

        for epoch in range(1, self.params['epochs']+1):
            epoch_loss = self._train_iter(epoch)
            training_losses = np.append(training_losses, epoch_loss)

            with torch.no_grad():
                eval_acc = self._evaluate()
                eval_accs = np.append(eval_accs, eval_acc)
                print(f"Epoch {epoch} LFW eval acc: {eval_acc:.4f}")
                self.scheduler.step(eval_acc)

            self._save_weights(epoch)
            self._save_plots(training_losses, eval_accs)
            self._log_csv(training_losses, eval_accs)


    def _get_lr(self):
        lr = [f"{group['lr']:.2e}" for group in self.optimizer.param_groups]
        return lr


    def _train_iter(self, curr_epoch):
        self.net.train()
        self.head.train()

        pbar = tqdm(self.train_loader,
                    mininterval=2.0,
                    position=0,
                    leave=True,
                    dynamic_ncols=True)
        losses = np.array([])
        for it, data in enumerate(pbar, 1):
            self.optimizer.zero_grad()

            # FORWARD PASS
            inputs, labels = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            feature = self.net(inputs)
            output = self.head(feature, labels)

            loss = self.criterion(output, labels)

            # BACKWARD PASS
            loss.backward()
            self.optimizer.step()

            losses = np.append(losses, loss.item())

            curr_lr = self._get_lr()
            avg_loss = np.mean(losses)

            desc = (f"[Ep {curr_epoch}][Iter {it}][lr {curr_lr}]" +
                    f"[AvgLoss: {avg_loss:.2f}]")

            pbar.set_description(desc, refresh=False)

        pbar.close()

        epoch_loss = np.mean(losses)
        return epoch_loss


    def _evaluate(self):
        self.net.eval()
        self.head.eval()

        lfw_dataset = LFWMasked('datasets/eval/data/lfw',
                                transform=self.eval_transform)

        evalloader = torch.utils.data.DataLoader(
            lfw_dataset,
            batch_size=self.params['batch_size'],
            num_workers=self.params['workers'],
            shuffle=False,
            pin_memory=False
        )

        features = {}
        pbar = tqdm(evalloader,
                    mininterval=2.0, position=0, leave=True,
                    dynamic_ncols=True)
        for imgs, keys in pbar:
            imgs = imgs.to(self.device)
            output = self.net(imgs)
            output = output.cpu().numpy()
            for i, key in enumerate(keys):
                features[str(key)] = output[i]

            pbar.set_description("lfw eval", refresh=False)
        pbar.close()

        preds = np.array([])
        trues = np.array([])

        pairs_file = list(lfw_dataset.pairs_file)
        for pair in pairs_file:
            feat1 = features[str(pair[0])]
            feat2 = features[str(pair[1])]
            ground_truth = int(pair[2])

            pred = calculate_cosin_sim(feat1, feat2)
            preds = np.append(preds, pred)
            trues = np.append(trues, ground_truth)

        _, _, _, threshold = roc_auc_threshold(preds, trues)
        acc = calculate_acc(preds, trues, threshold)

        return acc


    def _save_weights(self, epoch):
        save_loc = Path(WEIGHT_SAVE_DIR) / self.params['name'] / f"epoch_{epoch}"
        save_loc.mkdir(parents=True, exist_ok=True)

        save_net_name = f"{self.params['name']}_ep{epoch}.pth"
        save_net_name = str(save_loc / save_net_name)

        save_head_name = f"{self.params['name']}_head_ep{epoch}.pth"
        save_head_name = str(save_loc / save_head_name)

        if self.params['multi_gpu']:
            torch.save(self.net.module.state_dict(), save_net_name)
            torch.save(self.head.module.state_dict(), save_head_name)
        else:
            torch.save(self.net.state_dict(), save_net_name)
            torch.save(self.head.state_dict(), save_head_name)


    def _save_plots(self, training_losses, eval_accs):
        save_loc = Path(GRAPHS_SAVE_DIR) / self.params['name']
        save_loc.mkdir(parents=True, exist_ok=True)

        epochs = list(range(1, len(training_losses)+1))
        best_loss = np.argmin(training_losses)
        best_eval = np.argmax(eval_accs)

        plt.style.use('classic')

        plt.figure()
        plt.plot(epochs, training_losses, color='darkorange', lw=2)
        plt.plot([best_loss+1], [training_losses[best_loss]], marker="X", lw=5)
        plt.grid(b=True, which='major', axis='x',
                 color='#666666', linestyle='-', alpha=0.6)
        plt.xticks(epochs, fontsize=9)
        plt.xlabel(f'Epoch\nBest: {best_loss+1}')
        plt.ylabel('Train Loss')
        plt.title('Training statistics')
        plt.savefig(f"{str(save_loc)}/{self.params['name']}_training_loss.png")

        plt.figure()
        plt.plot(epochs, eval_accs, color='darkorange', lw=2)
        plt.plot([best_eval+1], [eval_accs[best_eval]], marker="X", lw=5)
        plt.grid(b=True, which='major', axis='x',
                 color='#666666', linestyle='-', alpha=0.6)
        plt.xticks(epochs, fontsize=9)
        plt.xlabel(f'Epoch\nBest: {eval_accs[best_eval]}@{best_eval+1}')
        plt.ylabel('Eval LFW Acc')
        plt.title('LFW eval statistics')
        plt.savefig(f"{str(save_loc)}/{self.params['name']}_eval_accs.png")
        plt.close('all')


    def _log_csv(self, training_losses, eval_accs):
        save_loc = Path(GRAPHS_SAVE_DIR) / self.params['name']
        save_loc.mkdir(parents=True, exist_ok=True)

        epochs = list(range(1, len(training_losses)+1))

        data = {
            'epoch': epochs,
            'loss': training_losses,
            'acc': eval_accs
        }

        log_df = pd.DataFrame(data=data)
        log_df.to_csv(f"{str(save_loc)}/{self.params['name']}_log.csv",
                      index=False)
