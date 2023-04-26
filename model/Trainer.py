# import necessary modules
import datetime
import json
import numpy as np
import os
import sys
import torch
import tqdm

from torch.nn import *
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from .utils import *


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        c_model, c_loss, c_optimizer, c_weights = config["MODEL"], config["LOSS"], \
                                                  config["OPTIMIZER"], config["MODEL"]["WEIGHTS"]

        # loss and optimizer attributes
        self.alpha = c_loss["SIMILARITY"]["weight"]     # NCC loss weight
        self.beta = c_loss["CONTRAST"]["weight"]        # HU loss weight
        self.gamma = c_loss["GAN"]["weight"]            # GAN loss weight
        self.bs = c_optimizer["batch_size"]             # batch size
        self.n_iter = c_optimizer["n_iter"]             # number of minibatches
        self.channels_in = c_model["GENERATOR"]["channels_in"]

        self.gen_every = c_optimizer["gen_every"]       # update generator every X iterations

        # set GAN architectures
        self.discriminator = self.str_to_class(c_model["DISCRIMINATOR"]["model"])(c_model["DISCRIMINATOR"])
        self.generator = self.str_to_class(c_model["GENERATOR"]["model"])(c_model["GENERATOR"])

        # set loss and optimizer
        self.lossSim = self.str_to_class(c_loss["SIMILARITY"]["name"])(c_loss["SIMILARITY"])
        self.lossHU = self.str_to_class(c_loss["CONTRAST"]["name"])(c_loss["CONTRAST"])
        self.lossGAN = self.str_to_class(c_loss["GAN"]["name"])()

        self.optimD = self.str_to_class(c_optimizer["discriminator"])(self.discriminator.parameters(),
                                                                      lr=c_optimizer["disc_lr"],
                                                                      betas=(c_optimizer["disc_b1"],
                                                                             c_optimizer["disc_b2"]))
        self.optimG = self.str_to_class(c_optimizer["generator"])(self.generator.parameters(),
                                                                  lr=c_optimizer["gen_lr"],
                                                                  betas=(c_optimizer["gen_b1"],
                                                                         c_optimizer["gen_b2"]))

        if c_weights["load"]:
            path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                                "logs", c_weights["timestamp"])

            # continue training from checkpoint
            self.iteration = c_weights["iteration"]
            c_optimizer["lr_milestones"] = [milestone - self.iteration for milestone in c_optimizer["lr_milestones"]]

            # model weights
            weights_d = torch.load(os.path.join(path, "discriminator", str(c_weights["iteration"]).zfill(7) + ".pt"))
            weights_g = torch.load(os.path.join(path, "generator", str(c_weights["iteration"]).zfill(7) + ".pt"))
            self.discriminator.load_state_dict(weights_d)
            self.generator.load_state_dict(weights_g)

            # optimizer weights
            weights_d = torch.load(os.path.join(path, "discriminator_opt", str(c_weights["iteration"]).zfill(7) + ".pt"))
            weights_g = torch.load(os.path.join(path, "generator_opt", str(c_weights["iteration"]).zfill(7) + ".pt"))
            self.optimD.load_state_dict(weights_d)
            self.optimG.load_state_dict(weights_g)
        else:
            self.iteration = 0

        self.schedulerD = self.str_to_class(c_optimizer["lr_policy"])(self.optimD,
                                                                      milestones=c_optimizer["lr_milestones"],
                                                                      gamma=c_optimizer["lr_gamma"])
        self.schedulerG = self.str_to_class(c_optimizer["lr_policy"])(self.optimG,
                                                                      milestones=c_optimizer["lr_milestones"],
                                                                      gamma=c_optimizer["lr_gamma"])

    def get_batch(self, loader, iterator):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        return batch[0].to(self.device), batch[1].to(self.device)

    @staticmethod
    def log_loss(writer, loss, iteration, mode):
        writer.add_scalar(f"{mode}/loss_I", loss[0].item(), iteration)
        writer.add_scalar(f"{mode}/loss_G", loss[1].item(), iteration)
        writer.add_scalar(f"{mode}/loss_D", loss[2].item(), iteration)

    def log_images(self, writer, input, output, iteration, tag):
        inputs_map = {
            f"{tag}_inputs": input,
            f"{tag}_outputs": output
        }
        img_sources = {}
        for name, batch in inputs_map.items():
            if isinstance(batch, list) or isinstance(batch, tuple):
                for i, b in enumerate(batch):
                    img_sources[f'{name}{i}'] = b.data.cpu().numpy()
            else:
                img_sources[name] = batch.data.cpu().numpy()

        for name, batch in img_sources.items():
            for i in (0.25, 0.75):
                image = batch[int(len(batch) * i), batch.shape[1] // 2][None, :, :]
                image = np.clip((image * self.lossHU.factor - self.lossHU.bias), -260, 740)
                image = ((image + 260) / 1000)
                writer.add_image(name + f"/{i}", image, iteration)

    @staticmethod
    def str_to_class(classname):
        return getattr(sys.modules[__name__], classname)

    def fit(self, train, val):
        now = datetime.datetime.now()
        writer_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            "logs",
            f"{now.day}-{now.month}-{now.year}_"
            f"{str(now.hour).zfill(2)}-{str(now.minute).zfill(2)}-{str(now.second).zfill(2)}"
        )
        writer = SummaryWriter(log_dir=writer_dir)
        os.makedirs(os.path.join(writer_dir, "generator"))
        os.makedirs(os.path.join(writer_dir, "discriminator"))

        # save config
        config_file = os.path.join(writer_dir, "config.json")
        with open(config_file, "w") as outfile:
            json.dump(self.config, outfile)

        train_low = DataLoader(train[0], batch_size=self.bs // 2, shuffle=True, num_workers=0)
        train_opt = DataLoader(train[1], batch_size=self.bs, shuffle=True, num_workers=0)
        train_high = DataLoader(train[2], batch_size=self.bs // 2, shuffle=True, num_workers=0)
        iter_low = iter(train_low)
        iter_opt = iter(train_opt)
        iter_high = iter(train_high)

        val_low = DataLoader(val[0], batch_size=2, shuffle=True, num_workers=0)
        val_opt = DataLoader(val[1], batch_size=2, shuffle=True, num_workers=0)
        val_high = DataLoader(val[2], batch_size=2, shuffle=True, num_workers=0)
        val_loaders = [val_low, val_opt, val_high]

        # to cuda
        self.discriminator.to(self.device)
        self.generator.to(self.device)

        self.discriminator.train()
        self.generator.train()
        for iteration in tqdm.tqdm(range(self.iteration, self.n_iter)):
            self.optimD.zero_grad()

            # get batch
            opt, opt_mask = self.get_batch(train_opt, iter_opt)
            low, low_mask = self.get_batch(train_low, iter_low)
            high, high_mask = self.get_batch(train_high, iter_high)

            subopt = torch.cat((low, high))
            subopt_mask = torch.cat((low_mask, high_mask))
            opt_hat = subopt - self.generator(subopt)

            # discriminator
            loss_d = self.gamma * self.lossGAN(self.discriminator(opt_hat.detach()), self.discriminator(opt))
            loss_d.backward()
            self.optimD.step()
            for p in self.discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)

            # generator
            if iteration % self.gen_every == 0:
                self.optimG.zero_grad()
                loss_a = -self.lossGAN(self.discriminator(opt_hat))
                loss_i = self.lossSim(opt_hat, subopt)
                loss_hu = self.lossHU(opt_hat, subopt_mask)

                if torch.isnan(loss_hu):
                    loss_g = self.gamma * loss_a + self.alpha * loss_i
                else:
                    loss_g = self.gamma * loss_a + self.alpha * loss_i + self.beta * loss_hu
                loss_g.backward()
                self.optimG.step()

            # update schedulers
            self.schedulerG.step()
            self.schedulerD.step()

            self.log_loss(writer, [loss_i, loss_a, loss_d], iteration, "train")
            if iteration % 200 == 0:
                self.log_images(writer, subopt, opt_hat, iteration, "images_train/fake")
                self.log_images(writer, opt, opt, iteration, "images_train/real")

                # evaluate
                self.discriminator.eval()
                self.generator.eval()

                # validation loop
                loss_i = torch.zeros(1).float().to(self.device)
                loss_a = torch.zeros(1).float().to(self.device)
                loss_real = torch.zeros(1).float().to(self.device)
                loss_fake = torch.zeros(1).float().to(self.device)
                for i, val_loader in enumerate(val_loaders):
                    if i != 1:
                        # generator
                        for j, batch in enumerate(val_loader):
                            if isinstance(batch, list):
                                batch = batch[0]
                            batch = batch.to(self.device)
                            with torch.no_grad():
                                batch_hat = batch - self.generator(batch)
                                loss_a -= self.lossGAN(self.discriminator(batch_hat))
                                loss_i += self.lossSim(batch_hat, batch)
                                loss_fake += self.lossGAN(self.discriminator(batch_hat.detach()))

                            if i == 0 and j == 0:
                                self.log_images(writer, batch, batch_hat, iteration, "images_val/low")
                            if i == 2 and j == 0:
                                self.log_images(writer, batch, batch_hat, iteration, "images_val/high")
                    else:
                        # discriminator
                        for j, batch in enumerate(val_loader):
                            if isinstance(batch, list):
                                batch = batch[0]
                            batch = batch.to(self.device)
                            with torch.no_grad():
                                loss_real -= self.lossGAN(self.discriminator(batch))

                            if j == 0:
                                self.log_images(writer, batch, batch, iteration, "images_val/real")

                loss_i /= (len(val_loaders[0]) + len(val_loaders[2]))
                loss_a /= (len(val_loaders[0]) + len(val_loaders[2]))
                loss_fake /= (len(val_loaders[0]) + len(val_loaders[2]))
                loss_real /= len(val_loaders[1])
                loss_d = (loss_real + loss_fake) / 2

                self.log_loss(writer, [loss_i, loss_a, loss_d], iteration, "val")
                self.discriminator.train()
                self.generator.train()

            if iteration % 1000 == 0:
                torch.save(self.generator.state_dict(),
                           os.path.join(writer_dir, "generator", str(iteration).zfill(7) + ".pt"))
                torch.save(self.discriminator.state_dict(),
                           os.path.join(writer_dir, "discriminator", str(iteration).zfill(7) + ".pt"))

                torch.save(self.optimG.state_dict(),
                           os.path.join(writer_dir, "generator_opt", str(iteration).zfill(7) + ".pt"))
                torch.save(self.optimD.state_dict(),
                           os.path.join(writer_dir, "discriminator_opt", str(iteration).zfill(7) + ".pt"))
