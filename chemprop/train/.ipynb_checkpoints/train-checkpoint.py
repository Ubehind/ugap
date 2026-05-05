from argparse import Namespace
import logging
from typing import Callable, List, Union
import os
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import trange

from chemprop.data import MoleculeDataset
from chemprop.nn_utils import compute_gnorm, compute_pnorm, NoamLR
from torch.distributions import Dirichlet, kl_divergence
import torch.nn.functional as F
from chemprop.features import mol2graph

import torch.nn.functional as F
import numpy as np
from chemprop.utils import compute_pearson_correlation, compute_spearman_correlation
from .plot_method import plot_binned_boxplot, visualize_mol_perturbation



def edl_multitask_loss(alpha, targets, numtasks, epoch=None, annealing_step=None, lam=1.0):

    targets = targets.to(torch.long)
    alpha = alpha.reshape((alpha.shape[0], numtasks, 2))
    B, T, C = alpha.shape
    S = torch.sum(alpha, dim=-1, keepdim=True)  # [B, T, 1]

    mask = (targets != -1).float()  # [B, T]

    # ---- One-hot 
    targets_clamped = torch.clamp(targets, min=0)
    y = F.one_hot(targets_clamped, num_classes=C).float()  # [B, T, C]
    loglikelihood = torch.sum(y * (torch.digamma(alpha) - torch.digamma(S)), dim=-1)
    loss_ce = -torch.sum(mask * loglikelihood) / (torch.sum(mask) + 1e-8)
    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = kl_divergence(Dirichlet(kl_alpha), Dirichlet(torch.ones(C, device='cuda:0')))
    kl_div = torch.sum(mask * kl_div) / (torch.sum(mask) + 1e-8)
    annealing_coef = min(1.0, epoch / annealing_step)
    loss = loss_ce + annealing_coef * kl_div
    loss = loss_ce + annealing_coef * lam * kl_div
    return loss, loss_ce.item(), kl_div.item(), alpha.detach()
def compute_vacuity_from_alpha(alpha):
    C = alpha.shape[-1]
    S = alpha.sum(dim=-1)
    vacuity = C / (S + 1e-8)
    return vacuity

def minmax_norm(x, dim=None, eps=1e-8):
    xmin = x.min(dim=dim)[0] if dim is not None else x.min()
    xmax = x.max(dim=dim)[0] if dim is not None else x.max()
    if dim is not None:
        xmin = xmin.unsqueeze(dim)
        xmax = xmax.unsqueeze(dim)
    return (x - xmin) / (xmax - xmin + eps)

def dirichlet_mean(alpha):
    S = alpha.sum(dim=-1, keepdim=True)
    return alpha / S
def kl_between_probs(p, q):
    eps = 1e-12
    return torch.sum(p * (torch.log(p + eps) - torch.log(q + eps)), dim=-1)


def evidential_loss_new(mu, v, alpha, beta, targets, lam=1, epoch=None, annealing_step=None, epsilon=1e-4):

    # Calculate NLL loss
    twoBlambda = 2*beta*(1+v)
    nll = 0.5*torch.log(np.pi/v) \
        - alpha*torch.log(twoBlambda) \
        + (alpha+0.5) * torch.log(v*(targets-mu)**2 + twoBlambda) \
        + torch.lgamma(alpha) \
        - torch.lgamma(alpha+0.5)

    L_NLL = nll 
    error = torch.abs((targets - mu))
    reg = error * (2 * v + alpha)
    L_REG = reg 
    annealing_coef = min(lam, epoch / annealing_step)
    loss = L_NLL + annealing_coef * (L_REG - epsilon)

    return loss






def compute_l_monotonic(alpha_clean, alpha_adv, nu_clean=None, nu_adv=None, task='regression'):
    """
    实现单调性损失：确保扰动后证据强度不增加（即不确定性不减小）
    """
    if task == 'regression':
        # 回归任务中，总证据强度通常指 S = alpha + nu (或者对应的虚观测数)
        # 也可以直接用证据分布的集中度参数
        s_clean = (alpha_clean + nu_clean).detach()
        s_adv = alpha_adv + nu_adv
    else:
        # 分类任务中，总证据强度 S = sum(alpha_i)
        s_clean = alpha_clean.sum(dim=-1).mean(dim=1, keepdim=True).detach()  # [B,1]
        s_adv = alpha_adv.sum(dim=-1).mean(dim=1, keepdim=True)  # [B,1]

    # 计算 S_adv - S_clean
    # 我们希望 S_adv <= S_clean，即差值 <= 0
    # 如果差值 > 0，则产生惩罚
    diff = s_adv - s_clean
    l_mono = torch.mean(F.relu(diff))
    
    return l_mono




def scale_invariant_ranking_loss(delta, S, use_log_s=True, top_k_ratio=0.5):
    delta = delta.view(-1, 1)
    S = S.view(-1, 1)
    
    if use_log_s:
        S_val = torch.log(S + 1e-8)
    else:
        S_val = S
    delta_diff = delta - delta.t()  # [B, B]
    S_diff = S_val.t() - S_val      # [B, B]
    base_mask = (delta_diff > 1e-6)


    if base_mask.any():

        diff_values = delta_diff[base_mask]

        k = int(diff_values.numel() * top_k_ratio)
        if k > 0:

            threshold = torch.topk(diff_values, k).values[-1]

            final_mask = base_mask & (delta_diff >= threshold)
        else:
            final_mask = base_mask
    else:
        final_mask = base_mask
    loss = F.softplus(-S_diff)
    valid_pairs = final_mask.sum()
    if valid_pairs > 0:
        total_loss = (loss * final_mask.float()).sum() / valid_pairs
    else:
        total_loss = S.sum() * 0.0 

    return total_loss

def self_correction_loss(S_clean, S_adv):
    log_S_clean = torch.log(S_clean.detach() + 1e-8)
    log_S_adv = torch.log(S_adv + 1e-8)
    l_self = torch.mean(F.relu(log_S_adv - log_S_clean))
    return l_self


def train(model: nn.Module,
          data: Union[MoleculeDataset, List[MoleculeDataset]],
          loss_func: Callable,
          optimizer: Optimizer,
          scheduler: _LRScheduler,
          args: Namespace,
          n_iter: int = 0,
          logger: logging.Logger = None,
          writer: SummaryWriter = None,
          epoch=None,
          scaler=None) -> int:
    """
    Trains a model for an epoch.

    :param model: Model.
    :param data: A MoleculeDataset (or a list of MoleculeDatasets if using moe).
    :param loss_func: Loss function.
    :param optimizer: An Optimizer.
    :param scheduler: A learning rate scheduler.
    :param args: Arguments.
    :param n_iter: The number of iterations (training examples) trained on so far.
    :param logger: A logger for printing intermediate results.
    :param writer: A tensorboardX SummaryWriter.
    :return: The total number of iterations (training examples) trained on so far.
    """
    count = 0
    debug = logger.debug if logger is not None else print
    
    model.train()
    
    data.shuffle()

    loss_sum, iter_count = 0, 0

    num_iters = len(data) // args.batch_size * args.batch_size  # don't use the last batch if it's small, for stability

    iter_size = args.batch_size

    for i in range(0, num_iters, iter_size):
        if i + args.batch_size > len(data):
            break
        mol_batch = MoleculeDataset(data[i:i + args.batch_size])
        smiles_batch, features_batch, target_batch = mol_batch.smiles(), mol_batch.features(), mol_batch.targets()
        batch = mol2graph(smiles_batch, args)
        mask = torch.Tensor([[x is not None for x in tb] for tb in target_batch])
        targets = torch.Tensor([[-1 if x is None else x for x in tb] for tb in target_batch])

        if next(model.parameters()).is_cuda:
            mask, targets = mask.cuda(), targets.cuda()

        class_weights = torch.ones(targets.shape)

        if args.cuda:
            class_weights = class_weights.cuda()
        model.zero_grad()
        output = model(batch, features_batch)
        x_emb = output['x_emb']
        x_emb.requires_grad_(True)
        x_emb.retain_grad()
        if args.dataset_type == 'classification':
            loss_clean, loss_ce, kl_div, alpha_clean = edl_multitask_loss(output['alphas'],targets, args.output_size, epoch=epoch, annealing_step=args.annealing_step, lam=args.lam)
            # loss_clean = loss_func(output['logits'], targets)
            grads = torch.autograd.grad(
                outputs=loss_clean,
                inputs=x_emb,
                retain_graph=True,   
                create_graph=False,  
                only_inputs=True
            )[0][1:]
            sens = grads.norm(p=2, dim=-1).detach()
            # sens = output['sens'][1:].detach()
            vacuity = compute_vacuity_from_alpha(output['alphas'])
            u_sample = vacuity.mean(dim=1, keepdim=True)
            u_norm = minmax_norm(u_sample, dim=0) 
            eps_max = args.eps_max
            eps_sample = (1.0 - u_norm) * eps_max
            N_nodes = x_emb.shape[0]
            node_delta = torch.zeros_like(x_emb, device='cuda:0')  # [N_nodes, emb_dim]
            batch_idx = batch.batch.to('cuda:0')
            batch_idx = batch_idx[1:]-1
            topk_ratio = args.topk
            for mol_id in torch.unique(batch_idx):
                node_mask = (batch_idx == mol_id)
                node_inds = node_mask.nonzero(as_tuple=False).view(-1)  # indices in 0..N_nodes-1
                if len(sens.shape)>1:
                    sens_m = sens[mol_id]
                    index = sens_m.nonzero().view(-1)
                    sens_m=sens_m[index]
                else:
                    sens_m = sens[node_inds]
                num_nodes = sens_m.size(0)
                k = max(1, int(max(1, num_nodes * topk_ratio)))  # ensure at least one
                # topk per molecule
                topk_local = torch.topk(sens_m, k=k, largest=True).indices
                node_sel = node_inds[topk_local]  # global node indices selected
                eps_nodes = eps_sample[mol_id].view(1)  # scalar -> [1]
                node_delta[1:][node_sel] = eps_nodes * grads[node_sel].sign()

            emb_adv = (x_emb.detach() + node_delta).detach()
            output_adv = model.forward_with_emb(batch, features_batch, emb_adv)
            loss_adv, loss_ce, kl_div, alpha_adv = edl_multitask_loss(output_adv['alphas'],targets, args.output_size, epoch=epoch, annealing_step=args.annealing_step, lam=args.lam)
            # loss_adv = loss_func(output_adv['logits'], targets)
            S_clean = output['alphas'].sum(dim=-1).mean(dim=1, keepdim=True)  # [B,1]
            S_adv = output_adv['alphas'].sum(dim=-1).mean(dim=1, keepdim=True)
            probs_clean = dirichlet_mean(alpha_clean)  # [B,T,C]
            probs_adv = dirichlet_mean(alpha_adv)
            kl_task = kl_between_probs(probs_clean, probs_adv)  # [B,T]
            delta = kl_task.mean(dim=1, keepdim=True).detach()        # [B,1]
            L_rank = scale_invariant_ranking_loss(delta, S_clean)
            L_self = self_correction_loss(S_clean, S_adv)
            # ---- 10) total loss and optimization ----
            beta =0
            if epoch > args.annealing_step:
                beta = args.beta
                # self_corr = args.self_corr
            total_loss = loss_clean + args.adv_w * loss_adv + beta * (L_rank + L_self)
            # total_loss = loss_adv
            # total_loss = loss_clean + args.adv_w * loss_adv
            # total_loss = loss_clean
        elif args.dataset_type == 'regression':
            loss_clean = evidential_loss_new(output['mu'], output['nu'], output['alpha'], output['beta'], targets, lam=args.lam, epoch=epoch, annealing_step=args.annealing_step).mean()
            # for dropout and ensemble experiments,use common loss such as MSE
            # loss_clean = loss_func(output['mu'], targets).mean()
            grads = torch.autograd.grad(
                outputs=loss_clean,
                inputs=x_emb,
                retain_graph=True,   
                create_graph=False,  
                only_inputs=True
            )[0][1:]
            sens = grads.norm(p=2, dim=-1).detach()
            # sens = output['sens'][1:]
            
            var = output['beta'] / (output['nu'] * (output['alpha'] - 1).clamp(min=1.0))
            u_norm = (var - var.min()) / (var.max() - var.min() + 1e-8)
            eps_max = args.eps_max
            eps_sample = (1.0 - u_norm) * eps_max

            N_nodes = x_emb.shape[0]
            node_delta = torch.zeros_like(x_emb, device='cuda:0') # [N_nodes, emb_dim]
            batch_idx = batch.batch.to('cuda:0')
            batch_idx = batch_idx[1:]-1
            topk_ratio = args.topk  #0.3
            for mol_id in torch.unique(batch_idx):

                node_mask = (batch_idx == mol_id)
                node_inds = node_mask.nonzero(as_tuple=False).view(-1)
                if len(sens.shape)>1:
                    sens_m = sens[mol_id]
                    index = sens_m.nonzero().view(-1)
                    sens_m=sens_m[index]
                else:
                    sens_m = sens[node_inds]
                num_nodes = sens_m.size(0)
                k = max(1, int(max(1, num_nodes * topk_ratio)))  # ensure at least one
                # topk per molecule
                topk_local = torch.topk(sens_m, k=k, largest=True).indices
                node_sel = node_inds[topk_local]  
                eps_nodes = eps_sample[mol_id].view(1)
                node_delta[1:][node_sel] = eps_nodes * grads[node_sel].sign()

            emb_adv = (x_emb.detach() + node_delta).detach()
            output_adv = model.forward_with_emb(batch, features_batch, emb_adv)
            loss_adv = evidential_loss_new(output_adv['mu'], output_adv['nu'], output_adv['alpha'], output_adv['beta'], targets, lam=args.lam, epoch=epoch, annealing_step=args.annealing_step).mean()
            # loss_adv = loss_func(output_adv['mu'], targets).mean()
            S_clean = output['alpha'] + output['nu']
            S_adv = output_adv['alpha'] + output_adv['nu']
            delta = torch.abs(output['mu'] - output_adv['mu']).detach()  # [B,1]
            L_rank = scale_invariant_ranking_loss(delta, S_clean, use_log_s=False)
            L_self = self_correction_loss(S_clean, S_adv)
            beta =0
            if epoch > args.annealing_step:
                beta = args.beta
            total_loss = loss_clean + args.adv_w * loss_adv + beta * (L_rank + L_self)
            # total_loss = loss_clean
        loss_sum += total_loss.item()

        iter_count += len(mol_batch)
        total_loss.backward()
        if args.clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_norm)
        optimizer.step()

        if isinstance(scheduler, NoamLR):
            scheduler.step()

        n_iter += len(mol_batch)

        # Log and/or add to tensorboard
        if (n_iter // args.batch_size) % args.log_frequency == 0:
            lrs = scheduler.get_lr()
            pnorm = compute_pnorm(model)
            gnorm = compute_gnorm(model)
            loss_avg = loss_sum / iter_count
            loss_sum, iter_count = 0, 0


            if writer is not None:
                writer.add_scalar('train_loss', loss_avg, n_iter)
                writer.add_scalar('param_norm', pnorm, n_iter)
                writer.add_scalar('gradient_norm', gnorm, n_iter)
                for i, lr in enumerate(lrs):
                    writer.add_scalar(f'learning_rate_{i}', lr, n_iter)

    return n_iter









def train_PGD(model: nn.Module,
          data: Union[MoleculeDataset, List[MoleculeDataset]],
          loss_func: Callable,
          optimizer: Optimizer,
          scheduler: _LRScheduler,
          args: Namespace,
          n_iter: int = 0,
          logger: logging.Logger = None,
          writer: SummaryWriter = None,
          epoch=None,
          scaler=None) -> int:
    model.train()
    data.shuffle()
    loss_sum, iter_count = 0, 0
    num_iters = len(data) // args.batch_size * args.batch_size
    iter_size = args.batch_size

    for i in range(0, num_iters, iter_size):
        if i + args.batch_size > len(data): break
        mol_batch = MoleculeDataset(data[i:i + args.batch_size])
        smiles_batch, features_batch, target_batch = mol_batch.smiles(), mol_batch.features(), mol_batch.targets()
        batch = mol2graph(smiles_batch, args)

        mask = torch.Tensor([[x is not None for x in tb] for tb in target_batch])
        targets = torch.Tensor([[-1 if x is None else x for x in tb] for tb in target_batch])
        if args.cuda: mask, targets = mask.cuda(), targets.cuda()
        model.zero_grad()
        output = model(batch, features_batch)
        x_emb = output['x_emb'] # [N_nodes, emb_dim]
        if args.dataset_type == 'classification':
            loss_clean, _, _, alpha_clean = edl_multitask_loss(output['alphas'], targets, args.output_size, epoch=epoch, annealing_step=args.annealing_step)
            u_sample = compute_vacuity_from_alpha(output['alphas']).mean(dim=1, keepdim=True)
        else:
            loss_clean = evidential_loss_new(output['mu'], output['nu'], output['alpha'], output['beta'], targets, lam=args.lam, epoch=epoch, annealing_step=args.annealing_step).mean()
            var = output['beta'] / (output['nu'] * (output['alpha'] - 1).clamp(min=1.0))
            u_sample = var 

        grads_clean = torch.autograd.grad(outputs=loss_clean, inputs=x_emb, retain_graph=True, only_inputs=True)[0]
        sens = grads_clean[1:].norm(p=2, dim=-1).detach() 
        
        u_norm = (u_sample - u_sample.min()) / (u_sample.max() - u_sample.min() + 1e-8)
        eps_sample = (1-torch.zeros_like(u_norm))*args.eps_max 
        batch_idx = batch.batch.to(x_emb.device)[1:] - 1
        node_sel_mask = torch.zeros(x_emb.shape[0], dtype=torch.bool, device=x_emb.device)
        for mol_id in torch.unique(batch_idx):
            node_mask = (batch_idx == mol_id)
            node_inds = node_mask.nonzero(as_tuple=False).view(-1)
            sens_m = sens[node_inds]
            num_nodes = sens_m.size(0)
            k = max(1, int(num_nodes * args.topk))
            topk_local = torch.topk(sens_m, k=k, largest=True).indices
            node_sel_mask[node_inds[topk_local] + 1] = True 

        pgd_steps = 5
        alpha_step = eps_sample / pgd_steps 
        
        delta = torch.zeros_like(x_emb).detach()

        full_mol_steps = torch.zeros((x_emb.shape[0], 1), device=x_emb.device)

        real_atom_batch_idx = batch.batch[1:].long() - 1 
        full_mol_steps[1:] = alpha_step[real_atom_batch_idx]
        
        for step in range(pgd_steps):
            delta.requires_grad = True
            emb_adv = x_emb.detach() + delta
            adv_out = model.forward_with_emb(batch, features_batch, emb_adv)
            
            if args.dataset_type == 'classification':
                l_adv, _, _, _ = edl_multitask_loss(adv_out['alphas'], targets, args.output_size, epoch=epoch, annealing_step=args.annealing_step)
            else:
                l_adv = evidential_loss_new(adv_out['mu'], adv_out['nu'], adv_out['alpha'], adv_out['beta'], targets, lam=args.lam, epoch=epoch, annealing_step=args.annealing_step).mean()
            
            grad_step = torch.autograd.grad(l_adv, delta)[0]
            
            with torch.no_grad():
                delta = delta + full_mol_steps * grad_step.sign()
                delta[~node_sel_mask] = 0

                full_eps_limit = torch.zeros((x_emb.shape[0], 1), device=x_emb.device)
                full_eps_limit[1:] = eps_sample[real_atom_batch_idx]
                
                delta = torch.clamp(delta, min=-full_eps_limit, max=full_eps_limit)
                delta = delta.detach()
        emb_adv_final = (x_emb.detach() + delta).detach()
        output_adv = model.forward_with_emb(batch, features_batch, emb_adv_final)
        
        if args.dataset_type == 'classification':
            loss_adv, _, _, alpha_adv = edl_multitask_loss(output_adv['alphas'], targets, args.output_size, epoch=epoch, annealing_step=args.annealing_step)
            S_clean = output['alphas'].sum(dim=-1).mean(dim=1, keepdim=True)
            S_adv = output_adv['alphas'].sum(dim=-1).mean(dim=1, keepdim=True)
            probs_clean = dirichlet_mean(alpha_clean)
            probs_adv = dirichlet_mean(alpha_adv)
            delta_y = kl_between_probs(probs_clean, probs_adv).mean(dim=1, keepdim=True).detach()
        else:
            loss_adv = evidential_loss_new(output_adv['mu'], output_adv['nu'], output_adv['alpha'], output_adv['beta'], targets, lam=args.lam, epoch=epoch, annealing_step=args.annealing_step).mean()
            S_clean = output['alpha'] + output['nu']
            S_adv = output_adv['alpha'] + output_adv['nu']
            delta_y = torch.abs(output['mu'] - output_adv['mu']).detach()

        L_rank = scale_invariant_ranking_loss(delta_y, S_clean, use_log_s=(args.dataset_type == 'classification'))
        L_self = self_correction_loss(S_clean, S_adv)

        beta = args.beta if epoch > args.annealing_step else 0
        
        total_loss = loss_clean + args.adv_w * loss_adv + beta * (L_rank + L_self)
        loss_sum += total_loss.item()
        iter_count += len(mol_batch)
        total_loss.backward()
        
        if args.clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_norm)
        optimizer.step()
        if isinstance(scheduler, NoamLR): scheduler.step()

        n_iter += len(mol_batch)
        
    return n_iter




def train_FLAG(model: nn.Module,
               data: Union[MoleculeDataset, List[MoleculeDataset]],
               loss_func: Callable,
               optimizer: Optimizer,
               scheduler: _LRScheduler,
               args: Namespace,
               n_iter: int = 0,
               logger: logging.Logger = None,
               writer: SummaryWriter = None,
               epoch=None,
               scaler=None) -> int:

    model.train()
    data.shuffle()
    loss_sum, iter_count = 0, 0
    num_iters = len(data) // args.batch_size * args.batch_size
    iter_size = args.batch_size

    m_steps = getattr(args, 'flag_m', 3) 
    step_size = args.eps_max / m_steps 

    for i in range(0, num_iters, iter_size):
        if i + args.batch_size > len(data): break
        mol_batch = MoleculeDataset(data[i:i + args.batch_size])
        smiles_batch, features_batch, target_batch = mol_batch.smiles(), mol_batch.features(), mol_batch.targets()
        batch = mol2graph(smiles_batch, args)

        targets = torch.Tensor([[-1 if x is None else x for x in tb] for tb in target_batch])
        if args.cuda: targets = targets.cuda()

        model.zero_grad()
        output_init = model(batch, features_batch)
        x_emb = output_init['x_emb'] # [N_nodes, emb_dim]

        delta = torch.zeros_like(x_emb).uniform_(-args.eps_max, args.eps_max).detach()
        delta.requires_grad = True

        for step in range(m_steps):
            emb_adv = x_emb.detach() + delta
            adv_out = model.forward_with_emb(batch, features_batch, emb_adv)
            

            if args.dataset_type == 'classification':
                loss, _, _, _ = edl_multitask_loss(adv_out['alphas'], targets, args.output_size, epoch=epoch, annealing_step=args.annealing_step)
            else:
                loss = evidential_loss_new(adv_out['mu'], adv_out['nu'], adv_out['alpha'], adv_out['beta'], targets, lam=args.lam, epoch=epoch, annealing_step=args.annealing_step).mean()

            loss_to_back = loss / m_steps
            loss_to_back.backward()
            if step < m_steps - 1:
                grad_delta = delta.grad.detach()
                delta.data = delta.data + step_size * torch.sign(grad_delta)
                delta.data = torch.clamp(delta.data, -args.eps_max, args.eps_max)
                delta.grad.zero_()
            loss_sum += loss_to_back.item()
        if args.clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_norm)
        optimizer.step()
        if isinstance(scheduler, NoamLR): scheduler.step()
        iter_count += len(mol_batch)
        n_iter += len(mol_batch)
    return n_iter















def train_CAP(model: nn.Module,
              data: Union[MoleculeDataset, List[MoleculeDataset]],
              loss_func: Callable,
              optimizer: Optimizer,
              scheduler: _LRScheduler,
              args: Namespace,
              n_iter: int = 0,
              logger: logging.Logger = None,
              writer: SummaryWriter = None,
              epoch: int = 0,
              scaler=None) -> int:

    model.train()
    data.shuffle()
    loss_sum, iter_count = 0, 0
    num_iters = len(data) // args.batch_size * args.batch_size
    iter_size = args.batch_size


    m_steps = getattr(args, 'cap_m', 3)       
    freq = getattr(args, 'cap_freq', 2)   

    is_feature_perturb = (epoch % freq == 0) if epoch is not None else False

    for i in range(0, num_iters, iter_size):
        if i + args.batch_size > len(data): break
        mol_batch = MoleculeDataset(data[i:i + args.batch_size])
        smiles_batch, features_batch, target_batch = mol_batch.smiles(), mol_batch.features(), mol_batch.targets()
        batch = mol2graph(smiles_batch, args)

        targets = torch.Tensor([[-1 if x is None else x for x in tb] for tb in target_batch])
        if args.cuda: targets = targets.cuda()

        if is_feature_perturb:

            step_size = args.eps_max / m_steps
            model.zero_grad()
            output_init = model(batch, features_batch)
            x_emb = output_init['x_emb']
            
            delta = torch.zeros_like(x_emb).uniform_(-args.eps_max, args.eps_max).detach()
            delta.requires_grad = True
            for step in range(m_steps):
                emb_adv = x_emb.detach() + delta
                adv_out = model.forward_with_emb(batch, features_batch, emb_adv)
                if args.dataset_type == 'classification':
                    loss, _, _, _ = edl_multitask_loss(adv_out['alphas'], targets, args.output_size, epoch=epoch, annealing_step=args.annealing_step)
                else:
                    loss = evidential_loss_new(adv_out['mu'], adv_out['nu'], adv_out['alpha'], adv_out['beta'], targets, lam=args.lam, epoch=epoch, annealing_step=args.annealing_step).mean()
                loss_to_back = loss / m_steps
                loss_to_back.backward()
                if step < m_steps - 1:
                    grad_delta = delta.grad.detach()
                    delta.data = delta.data + step_size * torch.sign(grad_delta)
                    delta.data = torch.clamp(delta.data, -args.eps_max, args.eps_max)
                    delta.grad.zero_()
                loss_sum += loss_to_back.item()
        else:
            model.zero_grad()
            output = model(batch, features_batch)
            if args.dataset_type == 'classification':
                loss_clean, _, _, _ = edl_multitask_loss(output['alphas'], targets, args.output_size, epoch=epoch, annealing_step=args.annealing_step)
            else:
                loss_clean = evidential_loss_new(output['mu'], output['nu'], output['alpha'], output['beta'], targets, lam=args.lam, epoch=epoch, annealing_step=args.annealing_step).mean()
            loss_clean.backward()
            weight_eps = getattr(args, 'weight_eps', 0.01) 
            original_params = {name: p.clone() for name, p in model.named_parameters() if p.grad is not None}
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if p.grad is not None:

                        grad_norm = torch.norm(p.grad) + 1e-12
                        p.add_(p.grad, alpha=(weight_eps / grad_norm)) 
            model.zero_grad()
            adv_out = model(batch, features_batch)
            if args.dataset_type == 'classification':
                loss_adv, _, _, _ = edl_multitask_loss(adv_out['alphas'], targets, args.output_size, epoch=epoch, annealing_step=args.annealing_step)
            else:
                loss_adv = evidential_loss_new(adv_out['mu'], adv_out['nu'], adv_out['alpha'], adv_out['beta'], targets, lam=args.lam, epoch=epoch, annealing_step=args.annealing_step).mean()
            loss_adv.backward()
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name in original_params:
                        p.copy_(original_params[name])
            loss_sum += loss_adv.item()
        if args.clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_norm)
        optimizer.step()
        if isinstance(scheduler, NoamLR): scheduler.step()
        iter_count += len(mol_batch)
        n_iter += len(mol_batch)
    return n_iter