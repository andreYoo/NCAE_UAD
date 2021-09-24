from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable
from utils.visualization.plot_images_grid import plot_images_grid,feature_plotting,TSNE_feature_plotting,entropy_based_plotting,plot_multiple_images_grid,TSNE_distributions_plotting,image_scatter,error_bar,image_scatter_with_coloured_boundary


import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import pdb


class NCAETrainer(BaseTrainer):

    def __init__(self, optimizer_name: str = 'adam', lr: float = 0.001 ,gan_lr: float = 0.0002, std:float=1.0,lamdba:float=0.1, n_epochs: int = 150, lr_milestones: tuple = (),
                 batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda', n_jobs_dataloader: int = 0,normal_cls: int =0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None
        self.normal_cls = normal_cls
        self.lamdba=lamdba
        self.batch_size =128
        self.std = std
        self.gan_lr = gan_lr
        self.lr = lr
        self.topk = int(self.batch_size*0.0)
        self.mu = None

    def train(self, dataset: BaseADDataset, net: BaseNet,d_l, d_s,d_g,g, writter=None):
        logger = logging.getLogger()

        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Initial setup for Adversarial learning
        real_label = 1
        fake_label = 0

        # Set loss
        criterion = nn.MSELoss(reduction='none')
        #criterion_D = nn.BCELoss()
        criterion_D = nn.CrossEntropyLoss()

        # Set device
        net = net.to(self.device)
        netD_l = d_l.to(self.device)
        netD_S = d_s.to(self.device)


        if self.mu is None:
            logger.info('Initializing initial statistics...')
            self.mu = self.init_center_c(train_loader, net.encoder)
            self.std_mtx = torch.ones(self.mu.size(),device=self.device)*self.std
            self.idt_mtx = torch.ones(self.mu.size(),device=self.device)
            logger.info('Mu and Std initialized.')



        criterion = criterion.to(self.device)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        optimizer_d = optim.Adam(net.decoder.parameters(), lr=0.0005, betas=(0.5, 0.999))
        optimizer_l = optim.Adam(netD_l.parameters(), lr=0.0005, betas=(0.5, 0.999))
        optimizer_s = optim.Adam(netD_S.parameters(), lr=0.0005, betas=(0.5, 0.999))
        #optimizer_g = optim.Adam(d_s.parameters(), lr=0.0001, betas=(0.5, 0.999))
        #0.0005 = 88.32
        #0.0001 = 9

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)
        scheduler_d = optim.lr_scheduler.MultiStepLR(optimizer_d, milestones=self.lr_milestones, gamma=0.1)
        scheduler_l = optim.lr_scheduler.MultiStepLR(optimizer_l, milestones=self.lr_milestones, gamma=0.1)
        scheduler_s = optim.lr_scheduler.MultiStepLR(optimizer_s, milestones=self.lr_milestones, gamma=0.1)
        #scheduler_G = optim.lr_scheduler.MultiStepLR(optimizer_G, milestones=self.lr_milestones, gamma=0.1)
        #scheduler_g = optim.lr_scheduler.MultiStepLR(optimizer_g, milestones=self.lr_milestones, gamma=0.1)

        # Training
        logger.info('Starting training...')
        start_time = time.time()
        net.train()
        for epoch in range(self.n_epochs):
            if epoch in self.lr_milestones:
                print('LR scheduler: new learning rate (for E,G, or Both) is %g' % float(scheduler.get_last_lr()[0]))
                print('LR scheduler: new learning rate (For D_L and D_S) is %g' % float(scheduler_l.get_last_lr()[0]))

            epoch_loss = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            mu_lr = scheduler.get_last_lr()[0]
            for _s, data in enumerate(train_loader):


                inputs, _, _, _ = data
                inputs = inputs.to(self.device)
                _refine_input = inputs.detach().clone()
                _inputs = inputs.detach().clone()
                gan_label = torch.LongTensor(inputs.size()[0]).fill_(0).cuda()

                ###########################
                # (1) Update Generator network (G)
                ###########################
                optimizer_d.zero_grad()
                # Original GAN loss
                for _ in range(inputs.size()[0]):
                    if _==0:
                        noise = torch.normal(self.mu,self.std_mtx).view(1,-1)
                    else:
                        noise = torch.cat((noise,torch.normal(self.mu,self.std_mtx).view( 1,-1)),0)
                #noise = torch.FloatTensor(inputs.size()[0], 32).normal_(self.mu, self.std).cuda()
                noise = Variable(noise)
                fake = net.decoder(noise)
                targetv = Variable(gan_label.fill_(real_label))
                output = netD_S(fake)
                output = output.squeeze()
                errG = criterion_D(output, targetv)
                errG.backward()
                errG_value = errG.item()
                optimizer_d.step()

                ###########################
                # (1) Update D_S network  #
                ###########################
                gan_label.fill_(real_label)
                targetv = Variable(gan_label)
                optimizer_s.zero_grad()
                output = netD_S(inputs)
                output = output.squeeze()
                errD_S_real = criterion_D(output, targetv)
                errD_S_real.backward()

                # noise = torch.FloatTensor(inputs.size()[0], 32).normal_(self.mu,1).cuda()
                # noise = Variable(noise)
                fake = net.decoder(noise)
                targetv = Variable(gan_label.fill_(fake_label))
                output = netD_S(fake.detach())
                output = output.squeeze()
                errD_S_fake = criterion_D(output, targetv)
                errD_S_fake.backward()
                errD_S_value = errD_S_real.item() + errD_S_fake.item()
                optimizer_s.step()

                ###########################
                # (1) Update Encoder network (f) + Generator (G)
                ###########################
                optimizer.zero_grad()
                # Original GAN loss
                targetv = Variable(gan_label.fill_(real_label))
                output = netD_l(noise)
                output = output.squeeze()
                errE = criterion_D(output, targetv)
                # errE.backward()

                clatent = net.encoder(fake.detach())
                fake_re = net.decoder(clatent.detach())

                rec,latent = net(inputs,get_latent=True)
                l_norm = latent.norm(p=2,dim=1,keepdim=True)
                c_norm = clatent.norm(p=2,dim=1,keepdim=True)
                latent_norm = latent.div(l_norm.expand_as(latent))
                clatent_norm = clatent.div(c_norm.expand_as(clatent))
                _gamma = torch.mean(latent_norm.mm(clatent_norm.t()),1)


                _, index_sorted = torch.sort(_gamma, dim=0, descending=False)
                rec_out = F.softmax(netD_S(rec))
                #pdb.set_trace()
                #inputs[index_sorted[0:self.topk]] = fake[index_sorted[0:self.topk]]
                rec_loss_1 = criterion(rec[index_sorted[self.topk:]], inputs[index_sorted[self.topk:]])
                rec_loss_2 = criterion(rec[index_sorted[0:self.topk]], fake[index_sorted[0:self.topk]])
                rec_loss = torch.cat((rec_loss_1,rec_loss_2),0)

                _refine_input[index_sorted[0:self.topk]] = fake[index_sorted[0:self.topk]]
                loss = torch.mean(rec_loss)
                total_loss = errE + self.lamdba * loss
                total_loss.backward()
                optimizer.step()

                ###########################
                # (1) Update D_L network    #
                ###########################
                gan_label.fill_(real_label)
                targetv = Variable(gan_label)
                optimizer_l.zero_grad()
                latent = net.encoder(inputs)
                output = netD_l(latent)
                output = output.squeeze()
                errD_l_real = criterion_D(output, targetv)
                errD_l_real.backward()


                for _ in range(inputs.size()[0]):
                    if _==0:
                        noise = torch.normal(self.mu,self.idt_mtx).view(1,-1)
                    else:
                        noise = torch.cat((noise,torch.normal(self.mu,self.idt_mtx).view(1,-1)),0)
                #noise = torch.FloatTensor(inputs.size()[0], 32).normal_(self.mu, self.std).cuda()
                noise = Variable(noise)
                targetv = Variable(gan_label.fill_(fake_label))
                output = netD_l(noise.detach())
                output = output.squeeze()
                errD_l_fake = criterion_D(output, targetv)
                errD_l_fake.backward()
                errD_l_value = errD_l_real.item() + errD_l_fake.item()
                optimizer_l.step()

                ###########################
                # (1) Update mu           #
                ###########################
                self.mu = self.mu-mu_lr*(torch.mean(self.mu-latent))

                ###########################
                self.mu = self.mu-mu_lr*(torch.mean(self.mu-latent))



                epoch_loss += total_loss.item()
                n_batches += 1
                if writter!=None:
                    writter.add_scalar("Train/D_s_Loss", errD_S_value, epoch * len(train_loader) + _s)
                    writter.add_scalar("Train/G_s_Loss", errG_value, epoch * len(train_loader) + _s)
                    writter.add_scalar("Train/D_l_Loss", errD_l_value, epoch * len(train_loader) + _s)
                    writter.add_scalar("Train/Recons_error", loss.item(), epoch * len(train_loader) + _s)
                    writter.add_scalar("Train/loss_total", total_loss.item(), epoch * len(train_loader) + _s)
                    writter.add_scalar("Train/Learning_rate", scheduler.get_last_lr()[0], epoch * len(train_loader) + _s)
                    writter.add_scalar("Train/Learning_rate_l", scheduler_l.get_last_lr()[0], epoch * len(train_loader) + _s)
                    writter.add_scalar("Train/Learning_rate_s", scheduler_s.get_last_lr()[0], epoch * len(train_loader) + _s)

            scheduler.step()
            scheduler_d.step()
            scheduler_l.step()
            scheduler_s.step()

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            if epoch%5==0:
                print(f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} | Train Time: {epoch_train_time:.3f}s '
                        f'| Train Loss: {epoch_loss / n_batches:.6f} |')

            if epoch%50==0 and epoch!=0:
                #plot_images_grid('../images/input_%d_epoch_%d_cls.png'%(epoch,self.normal_cls),inputs,title='Input images %d cls (%d)'%(self.normal_cls,epoch))
                #plot_images_grid('../images/recon_%d_epoch_%d_cls.png'%(epoch,self.normal_cls),rec,title='Reconstruction results %d cls (%d)'%(self.normal_cls,epoch))
                #plot_images_grid('../images/gen_%d_epoch_%d_cls.png'%(epoch,self.normal_cls),fake,title='Generated image %d cls (%d)'%(self.normal_cls,epoch))
                plot_multiple_images_grid('../images/%d_cls_%d_epoch.pdf'%(self.normal_cls,epoch),[_inputs,_refine_input,rec,fake,fake_re],title='%d class %d epoch images'%(self.normal_cls,epoch),subtitle=['input','refined_input','rec','gen','regen'])
                _2d_featires,_lbl  = TSNE_distributions_plotting('../images/%d_cls_%d_epoch_distributions.pdf'%(self.normal_cls,epoch),[latent.detach().cpu().numpy(),noise.detach().cpu().numpy()],leg=['Latent features','Sampled noise'])
                image_scatter_with_coloured_boundary('../images/%d_cls_%d_epoch_image_distributions.pdf'%(self.normal_cls,epoch),np.concatenate((inputs.detach().cpu().numpy(),fake.detach().cpu().numpy()),0),_2d_featires,_lbl)
                #error_bar('../images/%d_cls_%d_epoch_recon_discriminator_normality.png'%(self.normal_cls,epoch),rec.detach().cpu().numpy(),rec_out.detach().cpu().numpy()[:,0])
                #error_bar('../images/%d_cls_%d_epoch_recon_discriminator_abnormlaity.png' % (self.normal_cls, epoch), rec.detach().cpu().numpy(), rec_out.detach().cpu().numpy()[:,1])
                #pdb.set_trace()
                error_bar('../images/%d_cls_%d_epoch_recon_discriminator_cons_score.pdf' % (self.normal_cls, epoch), inputs.detach().cpu().numpy(), _gamma.detach().cpu().numpy())

        self.train_time = time.time() - start_time
        logger.info('Training Time: {:.3f}s'.format(self.train_time))
        logger.info('Finished training.')
        return net

    def test(self, dataset: BaseADDataset, ae_net: BaseNet):
        logger = logging.getLogger()

        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set loss
        criterion = nn.MSELoss(reduction='none')

        # Set device for network
        ae_net = ae_net.to(self.device)
        criterion = criterion.to(self.device)

        # Testing
        logger.info('Testing...')
        epoch_loss = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        ae_net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, _, idx = data
                inputs, labels, idx = inputs.to(self.device), labels.to(self.device), idx.to(self.device)

                rec = ae_net(inputs)
                rec_loss = criterion(rec, inputs)
                scores = torch.mean(rec_loss, dim=tuple(range(1, rec.dim())))

                # Save triple of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

                loss = torch.mean(rec_loss)
                epoch_loss += loss.item()
                n_batches += 1

        self.test_time = time.time() - start_time
        self.test_scores = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        self.test_auc = roc_auc_score(labels, scores)

        # Log results
        logger.info('[Experimental results]---------------------------------------------------------------------------')
        logger.info('Test Loss: {:.6f}'.format(epoch_loss / n_batches))
        logger.info('Test AUC: {:.2f}%'.format(100. * self.test_auc))
        logger.info('Test Time: {:.3f}s'.format(self.test_time))
        logger.info('Finished testing.')
        logger.info('================================================================================================')\

    def init_center_c(self, train_loader: DataLoader, encoder: BaseNet, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(encoder.rep_dim, device=self.device)

        encoder.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _, _, _ = data
                inputs = inputs.to(self.device)
                outputs = encoder(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c
