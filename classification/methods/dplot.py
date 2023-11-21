import os
import random
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.jit
import timm
import numpy as np 
import torchvision.transforms
from methods.base import TTAMethod
from sklearn.preprocessing import minmax_scale

import copy
from scipy import stats
from scipy.stats import shapiro, spearmanr
from copy import deepcopy
from datasets.data_loading import get_source_loader
from augmentations.transforms_cotta import get_tta_transforms

import logging
logger = logging.getLogger(__name__)
       
def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

def get_prototype(model, loader, num_classes):
    if type(model).__name__ == 'WideResNet':
        num_features = 640
    elif type(model).__name__ == 'Hendrycks2020AugMixResNeXtNet': 
        num_features = 1024
    elif type(model).__name__ == 'Kireev2021EffectivenessNet':
        num_features = 512
    elif type(model).__name__ == 'Hendrycks2020AugMixWRNNet':
        num_features = 128
    elif type(model).__name__ == 'Sequential':
        num_features = 2048
    features_proto = torch.zeros([num_classes, num_features])

    fs = []
    ys = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i > 500:
                break
            x, y = batch[0].cuda(), batch[1].cuda()

            if type(model).__name__ == 'WideResNet' or type(model).__name__ == 'Hendrycks2020AugMixWRNNet':
                out = model.conv1(x)
                out = model.block1(out)
                out = model.block2(out)
                out = model.block3(out)
                out = model.relu(model.bn1(out))
                out = F.avg_pool2d(out, 8)
                out = out.view(-1, model.nChannels)

            if type(model).__name__ == 'Kireev2021EffectivenessNet':
                out = model.conv1(x)
                out = model.layer1(out)
                out = model.layer2(out)
                out = model.layer3(out)
                out = model.layer4(out)
                if model.bn_before_fc:
                    out = F.relu(model.bn(out))
                out = F.avg_pool2d(out, 4)
                out = out.view(out.size(0), -1)

            if type(model).__name__ == 'Hendrycks2020AugMixResNeXtNet':
                x = model.conv_1_3x3(x)
                x = F.relu(model.bn_1(x), inplace=True)
                x = model.stage_1(x)
                x = model.stage_2(x)
                x = model.stage_3(x)
                x = model.avgpool(x)
                out = x.view(x.size(0), -1)
            
            if type(model).__name__ == 'Sequential':
                out = F.relu(model.model.bn1(model.model.conv1(x)))
                out = model.model.maxpool(out)
                out = model.model.layer1(out)
                out = model.model.layer2(out)
                out = model.model.layer3(out)
                out = model.model.layer4(out)
                out = F.avg_pool2d(out, 7)
                out = out.view(out.size(0), -1)

            ys.append(y)
            fs.append(out.cpu())
    ys = torch.cat(ys, dim=0).cpu()
    fs = torch.cat(fs, dim=0).cpu()

    for i in range(num_classes):
        features_proto[i] = fs[ys==i].mean(dim=0)
    # print(features_proto.shape)

    return features_proto

def protoytype_diffs(pre_proto, post_proto):
    diff = 0.

    for i in range(len(pre_proto)):
        diff += F.cosine_similarity(pre_proto[i].unsqueeze(0), post_proto[i].unsqueeze(0))

    return diff

def block_selection(model, src_loader, blocks, num_classes, alpha = 0.75):
    ce = nn.CrossEntropyLoss()
    selected_params = []
    best_blocks = []

    best_rate = 0.0

    test_model = copy.deepcopy(model)

    diff_list = []
    for block in blocks:
        selected_params = [block]
        # selected_params.append(block)

        test_model = copy.deepcopy(model)
        params = []     
        for name, m in test_model.named_parameters():
            if m.requires_grad and any([(x in name) for x in selected_params]):
                params.append(m)
        optimizer = optim.Adam(params, lr = 0.001)
        
        pre_prototype_vectors = get_prototype(test_model, src_loader, num_classes)
        # print(pre_prototype_vectors.shape)

        for i, batch in enumerate(src_loader):
            if i > 2000: 
                break
            imgs_src, labels_src = batch[0].cuda(), batch[1].cuda()
            outputs = test_model(imgs_src+ 0.5*torch.randn_like(imgs_src))

            loss = softmax_entropy(outputs).mean()
            loss.backward(retain_graph=False)

            optimizer.step()
            optimizer.zero_grad()

        post_prototype_vectors = get_prototype(test_model, src_loader, num_classes)
        
        diff = protoytype_diffs(pre_prototype_vectors, post_prototype_vectors)
        diff /= num_classes

        print(block, diff)
        diff_list.append(diff.item())

    diff_list = minmax_scale(diff_list)
    # print(diff_list)

    for i in range(len(diff_list)):
        print(blocks[i], diff_list[i])
        if diff_list[i]>alpha:
            best_blocks.append(blocks[i])
            
    print('# after optimization')
    return best_blocks


class DPLOT(TTAMethod):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)
        batch_size_src = 64 if num_classes == 1000 else 128
        _, self.src_loader = get_source_loader(dataset_name=cfg.CORRUPTION.DATASET,
                                               root_dir=cfg.DATA_DIR, adaptation=cfg.MODEL.ADAPTATION,
                                               batch_size=batch_size_src, ckpt_path=cfg.CKPT_PATH,
                                               percentage=cfg.SOURCE.PERCENTAGE,
                                               workers=min(cfg.SOURCE.NUM_WORKERS, os.cpu_count()))
        self.src_loader_iter = iter(self.src_loader)
        self.warmup_steps = cfg.RMT.NUM_SAMPLES_WARM_UP // batch_size_src
        self.final_lr = 0.001
        self.hflip = torchvision.transforms.functional.hflip
        
        self.criterion = nn.CrossEntropyLoss()
        self.cossim = nn.CosineSimilarity()
        if type(model).__name__ == 'WideResNet':
            blocks = ['block1.layer.0', 'block1.layer.1', 'block1.layer.2', 'block1.layer.3', 
                    'block2.layer.0', 'block2.layer.1', 'block2.layer.2', 'block2.layer.3', 
                    'block3.layer.0', 'block3.layer.1', 'block3.layer.2', 'block3.layer.3']


        if type(model).__name__ == 'Hendrycks2020AugMixWRNNet':
            blocks = ['block1.layer.0', 'block1.layer.1', 'block1.layer.2', 'block1.layer.3', 'block1.layer.4', 'block1.layer.5',
                    'block2.layer.0', 'block2.layer.1', 'block2.layer.2', 'block2.layer.3', 'block2.layer.4', 'block2.layer.5', 
                    'block3.layer.0', 'block3.layer.1', 'block3.layer.2', 'block3.layer.3', 'block3.layer.4', 'block3.layer.5']

        if type(model).__name__ == 'Kireev2021EffectivenessNet':
            blocks = ['layer1.0', 'layer1.1', 'layer2.0', 'layer2.1', 'layer3.0', 'layer3.1', 'layer4.0', 'layer4.1']

        if type(model).__name__ == 'Sequential':
            blocks = ['layer1.0', 'layer1.1', 'layer1.2', 
                        'layer2.0', 'layer2.1', 'layer2.2', 'layer2.3', 
                        'layer3.0', 'layer3.1', 'layer3.2', 'layer3.3', 'layer3.4', 'layer3.5', 
                        'layer4.0', 'layer4.1', 'layer4.2']

        if type(model).__name__ == 'Hendrycks2020AugMixResNeXtNet':
            blocks = ['stage_1.0', 'stage_1.1', 'stage_1.2', 'stage_2.0', 'stage_2.1', 'stage_2.2', 'stage_3.0', 'stage_3.1', 'stage_3.2']

        # blocks = block_selection(model, self.src_loader, blocks, num_classes)
        blocks = ['block1']

        # Setup EMA model
        self.model_ema = self.copy_model(self.model)
        for param in self.model_ema.parameters():
            param.detach_()
        self.warmup()

        print('Selected Blocks: ', blocks)
        params = []
        for block in blocks:
            for name, m in self.model.named_parameters():
                if m.requires_grad and (block in name):
                    params.append(m)
                    # print(name)
        self.optimizer = optim.Adam(params, lr = 0.001)
        self.ce_optimizer = optim.Adam(model.parameters(), lr = 0.0001)    

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        # print(x)
        imgs_test = x[0]
        imgs_test_hf = self.hflip(imgs_test)

        imgs = torch.cat([imgs_test, imgs_test_hf], dim=0)
        outputs = self.model(imgs) #

        with torch.no_grad():
            outputs_ema = self.model_ema(imgs)
            outputs_ema_avg = (outputs_ema[:len(imgs_test)] + outputs_ema[len(imgs_test):])/2

        ### First Stage 
        loss = softmax_entropy(outputs).mean()
        loss.backward(retain_graph=True)

        self.optimizer.step()
        self.model.zero_grad()
        
        ### Second Stage
        celoss = symmetric_cross_entropy(outputs[:len(imgs_test)], outputs_ema_avg).mean() + symmetric_cross_entropy(outputs[len(imgs_test):], outputs_ema_avg).mean()
        celoss.backward()

        self.ce_optimizer.step()
        self.model.zero_grad()

        self.model_ema = update_ema_variables(ema_model=self.model_ema, model=self.model, alpha_teacher=0.999)

        # Return ensemble predictions
        outputs = outputs + outputs_ema
        return outputs[:len(imgs_test)] + outputs[len(imgs_test):]

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def warmup(self):
        logger.info(f"Starting warm up...")
        for i in range(self.warmup_steps):
            #linearly increase the learning rate
            for par in self.optimizer.param_groups:
                par["lr"] = self.final_lr * (i+1) / self.warmup_steps

            # sample source batch
            try:
                batch = next(self.src_loader_iter)
            except StopIteration:
                self.src_loader_iter = iter(self.src_loader)
                batch = next(self.src_loader_iter)

            imgs_src, labels_src = batch[0], batch[1]
            imgs_src, labels_src = imgs_src.cuda(), labels_src.cuda().long()

            # forward the test data and optimize the model
            outputs = self.model(imgs_src)
            outputs_ema = self.model_ema(imgs_src)
            loss = symmetric_cross_entropy(outputs, outputs_ema).mean(0)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.model_ema = update_ema_variables(ema_model=self.model_ema, model=self.model, alpha_teacher=0.999)

        logger.info(f"Finished warm up...")
        for par in self.optimizer.param_groups:
            par["lr"] = self.final_lr

    def collect_params(self):
        """Collect the affine scale + shift parameters from batch norms.

        Walk the model's modules and collect all batch normalization parameters.
        Return the parameters and their names.

        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        params = list(self.model.parameters())
        return params, names

    def configure_model(self):
        """Configure model for use with tent."""
        # train mode, because tent optimizes the model to minimize entropy
        # self.model.train()
        self.model.eval()  # eval mode to avoid stochastic depth in swin. test-time normalization is still applied
        # disable grad, to (re-)enable only what tent updates
        # configure norm for tent updates: enable grad + force batch statisics
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()   # always forcing train mode in bn1d will cause problems for single sample tta
                m.requires_grad_(True)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)


@torch.jit.script
def softmax_entropy(x) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

@torch.jit.script
def symmetric_cross_entropy(x, x_ema):# -> torch.Tensor:
    return -0.5*(x_ema.softmax(1) * x.log_softmax(1)).sum(1)-0.5*(x.softmax(1) * x_ema.log_softmax(1)).sum(1)