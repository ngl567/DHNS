# coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import sys
import datetime
import ctypes
import json
import numpy as np
import copy
from tqdm import tqdm


class Trainer_dhns(object):

    def __init__(self,
                 model=None,
                 data_loader=None,
                 train_times=1000,
                 alpha=0.5,
                 use_gpu=True,
                 opt_method="sgd",
                 save_steps=None,
                 checkpoint_dir=None,
                 train_mode='adp',
                 beta=0.5,
                 generator=None,
                 lrg=None,
                 mu=None,
                 g_epoch=100):

        self.work_threads = 8
        self.train_times = train_times

        self.opt_method = opt_method
        self.optimizer = None
        self.lr_decay = 0
        self.weight_decay = 0
        self.alpha = alpha

        self.model = model
        self.data_loader = data_loader
        self.use_gpu = use_gpu
        self.save_steps = save_steps
        self.checkpoint_dir = checkpoint_dir

        self.train_mode = train_mode
        self.beta = beta

        # learning rate of the generator
        assert lrg is not None
        self.alpha_g = lrg

        # the generator part
        assert generator is not None
        assert mu is not None
        self.optimizer_g = None
        self.generator = generator
        self.batch_size = self.model.batch_size
        self.generator.cuda()
        self.mu = mu
        self.g_epoch = g_epoch

    def train_one_step(self, data):
        self.optimizer.zero_grad()
        loss, p_score = self.model({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': self.to_var(data['batch_r'], self.use_gpu),
            'batch_y': self.to_var(data['batch_y'], self.use_gpu),
            'mode': data['mode']
        })

        # training DHNE generator module
        batch_h_gen = self.to_var(data['batch_h'][0: self.batch_size], self.use_gpu)
        batch_t_gen = self.to_var(data['batch_t'][0: self.batch_size], self.use_gpu)
        batch_r = self.to_var(data['batch_r'][0: self.batch_size], self.use_gpu)
        batch_hs = self.model.model.get_batch_ent_embs(batch_h_gen)
        batch_ts = self.model.model.get_batch_ent_embs(batch_t_gen)
        batch_r = self.model.model.get_batch_rel_embs(batch_r)
        #print("batch_r:\t", batch_r.size())
        batch_hv = self.model.model.get_batch_img_embs(batch_h_gen)
        batch_tv = self.model.model.get_batch_img_embs(batch_t_gen)
        batch_ht = self.model.model.get_batch_text_embs(batch_h_gen)
        batch_tt = self.model.model.get_batch_text_embs(batch_t_gen)
        def train_diffusion():
            for epoch in range(self.g_epoch):
                self.optimizer_g.zero_grad()
                
                diff_loss = self.generator(batch_hs, batch_r, batch_ts) # structurl diffusion loss
                diff_loss += self.generator(batch_hv, batch_r, batch_tv) # visual diffusion loss
                diff_loss += self.generator(batch_ht, batch_r, batch_tt) # textual diffusion loss
                diff_loss.backward(retain_graph=True)
                self.optimizer_g.step()
                return diff_loss
        diff_loss = train_diffusion()

        # generate multimodal semantics-guided negative samples
        batch_neg_h, batch_neg_t = self.generator.sample(batch_hs, batch_r, batch_ts)
        batch_neg_hv, batch_neg_tv = self.generator.sample(batch_hv, batch_r, batch_tv)
        batch_neg_ht, batch_neg_tt = self.generator.sample(batch_ht, batch_r, batch_tt)

        # multi-level hard negative sample-based learning
        w = [0, 0.5, 1.0, 1.0, 0.5]
        w_m = [0.1, 0.3, 0.5, 0.7, 0.9]
        neg_list = []

        for i in range(len(w)):
            scores = self.model.model.mm_negative_score(
                batch_h=batch_h_gen,
                batch_r=batch_r,
                batch_t=batch_t_gen,
                mode=data['mode'],
                w_margin=w_m[i],
                neg_h=batch_neg_h[i],
                neg_t=batch_neg_t[i],
                neg_hv=batch_neg_hv[i],
                neg_tv=batch_neg_tv[i],
                neg_ht=batch_neg_ht[i],
                neg_tt=batch_neg_tt[i]
            )

            for score in scores:
                neg_list.append(self.model.loss(p_score, score) * w[i] * self.mu)
        sam = [1 for i in w if i != 0]
        loss_neg = sum(neg_list) / (sum(sam)*3)
        loss += loss_neg

        loss.backward()
        self.optimizer.step()
        return loss.item(), diff_loss.item()

    def run(self):
        if self.use_gpu:
            self.model.cuda()

        if self.optimizer is not None:
            pass
        elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
            self.optimizer = optim.Adagrad(
                self.model.parameters(),
                lr=self.alpha,
                lr_decay=self.lr_decay,
                weight_decay=self.weight_decay,
            )
            self.optimizer_g = optim.Adam(
                self.generator.parameters(),
                lr=self.alpha_g,
                weight_decay=self.weight_decay,
            )
        elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
            self.optimizer = optim.Adadelta(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
            self.optimizer_g = optim.Adam(
                self.generator.parameters(),
                lr=self.alpha_g,
                weight_decay=self.weight_decay,
            )
        elif self.opt_method == "Adam" or self.opt_method == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
            self.optimizer_g = optim.Adam(
                self.generator.parameters(),
                lr=self.alpha_g,
                weight_decay=self.weight_decay,
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
            self.optimizer_g = optim.Adam(
                self.generator.parameters(),
                lr=self.alpha_g,
                weight_decay=self.weight_decay,
            )
        print("Finish initializing...")

        training_range = tqdm(range(self.train_times))
        for epoch in training_range:
            res = 0.0
            res_g = 0.0
            for data in self.data_loader:
                loss, loss_g = self.train_one_step(data)
                res += loss
                res_g += loss_g
            training_range.set_description("Epoch %d | KGC loss: %f, DiffHEG loss %f" % (epoch, res, res_g))

            if self.save_steps and self.checkpoint_dir and (epoch + 1) % self.save_steps == 0:
                print("Epoch %d has finished, saving..." % (epoch))
                self.model.save_checkpoint(os.path.join(self.checkpoint_dir + "-" + str(epoch) + ".ckpt"))

    def set_model(self, model):
        self.model = model

    def to_var(self, x, use_gpu):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_lr_decay(self, lr_decay):
        self.lr_decay = lr_decay

    def set_weight_decay(self, weight_decay):
        self.weight_decay = weight_decay

    def set_opt_method(self, opt_method):
        self.opt_method = opt_method

    def set_train_times(self, train_times):
        self.train_times = train_times

    def set_save_steps(self, save_steps, checkpoint_dir=None):
        self.save_steps = save_steps
        if not self.checkpoint_dir:
            self.set_checkpoint_dir(checkpoint_dir)

    def set_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
