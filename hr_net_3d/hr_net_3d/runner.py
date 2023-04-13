import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from .hr_net_3d import HighResolutionNet
from .losses import CustomMSELoss, CustomCrossEntropyLoss

import numpy as np
import time
import datetime
import os


class Runner(object):

    def __init__(self, data_iter, params):
        """
        Constructor of the runner class

        Parameters
        ----------
        params:         List of hyper-parameters of the model

        Returns
        -------
        Initialises the dataloaders, the model, and the optimizers

        """
        self.p = params

        torch.cuda.set_device(int(self.p['gpu']))

        np.random.seed(self.p['seed'])
        torch.manual_seed(self.p['seed'])

        if self.p['gpu'] != '-1' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.cuda.set_rng_state(torch.cuda.get_rng_state())
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device('cpu')

        self.data_iter = data_iter

        self.p['out_chs'] = len(self.p["labels"]) + 1  # one extra channel for the background
        self.model = HighResolutionNet(out_chs=self.p['out_chs'], architecture=self.p.get("architecture", None))
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.p['lr'], weight_decay=self.p['l2'])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=3000, T_mult=2, eta_min=1e-5)

        self.mse_loss = CustomMSELoss(reduction="mean")
        self.cross_entropy_loss = CustomCrossEntropyLoss(reduction="mean")

        log_dir = os.path.join(params['log_dir'], 'runs/{}'.format(self.p['name']))

        self.writer = SummaryWriter(
            log_dir=os.path.join(log_dir, datetime.datetime.now().strftime("%B-%d-%Y--%I-%M%p")))

    def save_model(self, save_path):
        state = {
            'state_dict': self.model.state_dict(),
            'best_val': self.best_val,
            'best_epoch': self.best_epoch,
            'optimizer': self.optimizer.state_dict(),
            'args': self.p
        }
        torch.save(state, save_path)

    def load_model(self, load_path):

        state = torch.load(load_path, map_location=self.device)
        state_dict = state['state_dict']
        self.best_val = state['best_val']
        self.best_epoch = state['best_epoch']
        self.best_val_idrate = self.best_val['id_rate']

        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(state['optimizer'])

    def read_batch(self, batch, split):
        """
        Function to read a batch of data and move the tensors in batch to CPU/GPU

        Remember to transpose image to NCHW

        """
        if split == 'train':
            im, msk, cents = batch

            im = im.unsqueeze(1).to(self.device).type(torch.float32)  # convert NHW to NCHW

            msk = msk.permute(0, 4, 1, 2, 3).to(self.device).type(torch.float32)

            cents = cents.to(self.device).type(torch.float32)

            return im, msk, cents

        else:

            im, cents, im_hr, path = batch

            im = im.unsqueeze(1).to(self.device).type(torch.float32)  # convert NHW to NCHW
            im_hr = im_hr.unsqueeze(1).to(self.device).type(torch.float32)  # convert NHW to NCHW

            cents = cents.to(self.device).type(torch.float32)

            return im, cents, im_hr, path

    def run_epoch(self, epoch):
        """
        Function to run one epoch of training

        Parameters
        ----------
        epoch: current epoch count

        Returns
        -------
        loss: The loss value after the completion of one epoch
        """
        start = time.time()

        self.model.train()
        losses = []
        train_iter = iter(self.data_iter['train'])
        len_train_iter = len(self.data_iter['train'])

        for step, batch in enumerate(train_iter):
            self.optimizer.zero_grad()
            im, msk, _ = self.read_batch(batch, 'train')

            pred, _ = self.model.forward(im)

            _, _, h, w, d = im.shape
            pred = F.interpolate(pred, size=(h, w, d), mode='trilinear')

            # heatmap loss
            loss_mse = self.mse_loss(pred, msk)

            # heatmap cross entropy loss
            loss_ce = self.cross_entropy_loss(pred, msk)

            loss = loss_mse + self.p.get('ce_loss_weight', 1e-3) * loss_ce

            loss.backward()

            # write total loss
            self.writer.add_scalar("loss/train_loss_total", loss, step + len_train_iter * (epoch + 1))

            self.optimizer.step()
            losses.append(loss.item())

            if step % 10 == 0:
                print(f'[E:{epoch}| {step}]: Train Loss:{np.mean(losses):.5f}, Val IdRate:{self.best_val_idrate:.5}, '
                      f'\t{self.p["name"]}')

            self.scheduler.step(epoch + step / len_train_iter)

        loss = np.mean(losses)
        print('[Epoch:{}]:  Training Loss:{:.4}'.format(epoch, loss))
        print('Time:{0:.2f}s\n'.format(time.time() - start))

        return loss

    def fit(self):
        """
        Function to run training and evaluation of model

        Parameters
        ----------

        Returns
        -------
        """
        self.best_val_idrate, self.best_val, self.best_epoch, val_idrate = 0., {}, 0, 0.
        save_path = os.path.join(self.p['save_dir'], self.p['name'])

        if self.p.get('restore', False):
            self.load_model(save_path)
            print('[*] Successfully Loaded previous model')

        kill_cnt = 0
        for epoch in range(self.p['max_epochs']):
            start_time = time.time()
            train_loss = self.run_epoch(epoch)
            stop_time = time.time()
            val_results = self.evaluate('valid', epoch, plot=True)

            if val_results['id_rate'] > self.best_val_idrate:
                self.best_val = val_results
                self.best_val_idrate = val_results['id_rate']
                self.best_epoch = epoch
                self.save_model(save_path)
                kill_cnt = 0
            else:
                kill_cnt += 1
                if kill_cnt > self.p['max_kill_count']:
                    print("[*] Early Stopping!!")
                    break

            print(f'[Epoch {epoch}]: Training Loss: {train_loss:.5}, Valid IdRate: {self.best_val_idrate:.5} '
                  f'Time: {int(stop_time - start_time)}s\n\n')

    def predict(self, im):
        pred_3d, _ = self.model.forward(im)
        _, _, h, w, d = im.shape
        pred_3d = F.interpolate(pred_3d, size=(h, w, d), mode='trilinear')

        pred_3d = torch.softmax(pred_3d, 1)[:, 1:, ...]  # throw the background channel away
        pred_3d = pred_3d.permute(0, 2, 3, 4, 1)[0, ...]
        return pred_3d
