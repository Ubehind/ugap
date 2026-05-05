from typing import List
from argparse import Namespace
import torch
import torch.nn as nn
from tqdm import trange
import numpy as np

from chemprop.data import MoleculeDataset, StandardScaler
from chemprop.features import mol2graph


def predict(model: nn.Module,
            data: MoleculeDataset,
            batch_size: int,
            scaler: StandardScaler = None,
            args:Namespace = None):
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A model.
    :param data: A MoleculeDataset.
    :param batch_size: Batch size.
    :param scaler: A StandardScaler object fit on the training targets.
    :return: A list of lists of predictions. The outer list is examples
    while the inner list is tasks.
    """
    model.eval()

    preds = []
    conf = []
    alphas = []
    total_vars = []
    num_iters, iter_step = len(data), batch_size

    for i in range(0, num_iters, iter_step):
        # Prepare batch
        mol_batch = MoleculeDataset(data[i:i + batch_size])
        smiles_batch, features_batch = mol_batch.smiles(), mol_batch.features()
        batch = mol2graph(smiles_batch, args)
        # Run model
        # batch = smiles_batch

        with torch.no_grad():
            output = model(batch, features_batch)
            batch_preds = output['preds']

        batch_preds = batch_preds.data.cpu().numpy()
        if args.dataset_type == 'classification':
            batch_preds = batch_preds[:, :, 1]
            batch_conf = 1 - output['alphas'].shape[-1] / torch.sum(output['alphas'], dim=-1)
            # batch_conf = output['preds'].max(dim=-1)[0]
            batch_alphas = output['alphas'].tolist()
            alphas.extend(batch_alphas)
        else:
            # batch_conf = output['alpha']/ (output['alpha'] + output['beta'])
            batch_conf = output['beta'] / ((output['alpha']-1) * output['nu'])
            # batch_conf = torch.exp(-batch_conf)
            batch_conf = 1/batch_conf

            batch_total_vars = (output['beta'] / (output['alpha'] - 1)) * (1 + 1 / output['nu'])
            total_vars.extend(batch_total_vars.cpu().numpy().tolist())


        # Inverse scale if regression
        if scaler is not None:
            batch_preds = scaler.inverse_transform(batch_preds)

        # Collect vectors
        batch_preds = batch_preds.tolist()
        batch_conf = batch_conf.tolist()
        


        preds.extend(batch_preds)
        conf.extend(batch_conf)

    return preds, conf, alphas, total_vars




def predict_with_dropout(model: nn.Module,
                         data: MoleculeDataset,
                         batch_size: int,
                         num_samples: int = 10,
                         scaler: StandardScaler = None,
                         args: Namespace = None):


    model.eval()
    

    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

    preds = []
    conf = []
    alphas = [] 

    num_iters, iter_step = len(data), batch_size

    for i in range(0, num_iters, iter_step):

        mol_batch = MoleculeDataset(data[i:i + batch_size])
        smiles_batch, features_batch = mol_batch.smiles(), mol_batch.features()
        batch = mol2graph(smiles_batch, args)

        batch_sample_preds = []

        with torch.no_grad():

            for _ in range(num_samples):
                output = model(batch, features_batch)

                
                if args.dataset_type == 'classification':
                    batch_p = output['logits']

                    batch_p = torch.softmax(batch_p, dim=-1)[:, :, 1]
                else:
                    batch_p = output['preds']
                
                batch_sample_preds.append(batch_p)

        batch_sample_preds = torch.stack(batch_sample_preds)
        mean_preds = torch.mean(batch_sample_preds, dim=0)
        variance = torch.var(batch_sample_preds, dim=0)
        batch_conf = 1.0 / (variance + 1e-6)
        mean_preds_np = mean_preds.cpu().numpy()
        if scaler is not None:
            mean_preds_np = scaler.inverse_transform(mean_preds_np)

        preds.extend(mean_preds_np.tolist())
        conf.extend(batch_conf.cpu().numpy().tolist())
    return preds, conf, alphas


