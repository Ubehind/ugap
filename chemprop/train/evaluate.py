import logging
from typing import Callable, List

import torch.nn as nn

from .predict import predict, predict_with_dropout
from chemprop.data import MoleculeDataset, StandardScaler
from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from .plot_method import plot_cutoff, save_confidence_vs_rmse, plot_classification_experiments,save_quantile_conf_vs_accuracy,save_quantile_conf_vs_auc,analysis_calibration_ece,analysis_regression_calibration,plot_calibration_curve


def evaluate_predictions(preds: List[List[float]],
                         targets: List[List[float]],
                         num_tasks: int,
                         metric_func: Callable,
                         dataset_type: str,
                         logger: logging.Logger = None) -> List[float]:
    """
    Evaluates predictions using a metric function and filtering out invalid targets.

    :param preds: A list of lists of shape (data_size, num_tasks) with model predictions.
    :param targets: A list of lists of shape (data_size, num_tasks) with targets.
    :param num_tasks: Number of tasks.
    :param metric_func: Metric function which takes in a list of targets and a list of predictions.
    :param dataset_type: Dataset type.
    :param logger: Logger.
    :return: A list with the score for each task based on `metric_func`.
    """
    info = logger.info if logger is not None else print

    if len(preds) == 0:
        return [float('nan')] * num_tasks

    # Filter out empty targets
    # valid_preds and valid_targets have shape (num_tasks, data_size)
    valid_preds = [[] for _ in range(num_tasks)]
    valid_targets = [[] for _ in range(num_tasks)]
    for i in range(num_tasks):
        for j in range(len(preds)):
            if targets[j][i] is not None:  # Skip those without targets
                valid_preds[i].append(preds[j][i])
                valid_targets[i].append(targets[j][i])

    # Compute metric
    results = []
    for i in range(num_tasks):
        # # Skip if all targets or preds are identical, otherwise we'll crash during classification
        if dataset_type == 'classification':
            nan = False
            if all(target == 0 for target in valid_targets[i]) or all(target == 1 for target in valid_targets[i]):
                nan = True
                info('Warning: Found a task with targets all 0s or all 1s')
            if all(pred == 0 for pred in valid_preds[i]) or all(pred == 1 for pred in valid_preds[i]):
                nan = True
                info('Warning: Found a task with predictions all 0s or all 1s')

            if nan:
                results.append(float('nan'))
                continue

        if len(valid_targets[i]) == 0:
            continue

        if dataset_type == 'multiclass':
            results.append(metric_func(valid_targets[i], valid_preds[i], labels=list(range(len(valid_preds[i][0])))))
        else:
            results.append(metric_func(valid_targets[i], valid_preds[i]))

    return results


def evaluate(model: nn.Module,
             data: MoleculeDataset,
             num_tasks: int,
             metric_func: Callable,
             batch_size: int,
             dataset_type: str,
             scaler: StandardScaler = None,
             logger: logging.Logger = None,
             args: Namespace=None) -> List[float]:
    """
    Evaluates an ensemble of models on a dataset.

    :param model: A model.
    :param data: A MoleculeDataset.
    :param num_tasks: Number of tasks.
    :param metric_func: Metric function which takes in a list of targets and a list of predictions.
    :param batch_size: Batch size.
    :param dataset_type: Dataset type.
    :param scaler: A StandardScaler object fit on the training targets.
    :param logger: Logger.
    :return: A list with the score for each task based on `metric_func`.
    """
    preds,_,_,_ = predict(
        model=model,
        data=data,
        batch_size=batch_size,
        scaler=scaler,
        args=args
    )
    
    targets = data.targets()

    results = evaluate_predictions(
        preds=preds,
        targets=targets,
        num_tasks=num_tasks,
        metric_func=metric_func,
        dataset_type=dataset_type,
        logger=logger
    )

    return results






def evaluate_and_plot(model: nn.Module,
             data: MoleculeDataset,
             num_tasks: int,
             metric_func: Callable,
             batch_size: int,
             dataset_type: str,
             scaler: StandardScaler = None,
             logger: logging.Logger = None,
             args: Namespace=None) -> List[float]:
    """
    Evaluates an ensemble of models on a dataset.

    :param model: A model.
    :param data: A MoleculeDataset.
    :param num_tasks: Number of tasks.
    :param metric_func: Metric function which takes in a list of targets and a list of predictions.
    :param batch_size: Batch size.
    :param dataset_type: Dataset type.
    :param scaler: A StandardScaler object fit on the training targets.
    :param logger: Logger.
    :return: A list with the score for each task based on `metric_func`.
    """
    
    if args.use_dropout:
        preds,conf, alphas = predict_with_dropout(
            model=model,
            data=data,
            batch_size=batch_size,
            scaler=scaler,
            args=args
        )
    else:
        preds,conf, alphas, total_vars = predict(
            model=model,
            data=data,
            batch_size=batch_size,
            scaler=scaler,
            args=args
        )
    
    targets = data.targets()


    if args.dataset_type == 'regression':
        save_confidence_vs_rmse(targets, preds, conf)
        plot_calibration_curve(preds,targets,total_vars,save_path='images/regression_calibration.png', beta_val=args.rank_corr)
    else:
        save_quantile_conf_vs_accuracy(targets,preds,conf)
        save_quantile_conf_vs_auc(targets,preds,conf)
        analysis_calibration_ece(preds, conf,targets,n_bins=10,title="ece")






























def compute_conf(mu, nu, alpha, beta):
    """根据 NIG 参数计算置信度 [B,1] -> [B]"""
    # evidence = (alpha - 1) * nu   # [B,1]
    # conf = evidence / (1.0 + evidence)

    # 这是总不确定性
    # total_uncer = beta / (alpha - 1) + beta / (nu * (alpha - 1))
    # return total_uncer.squeeze(-1).cpu().numpy()

    # 这是置信度
    conf = alpha / (alpha + beta)
    return conf.squeeze(-1).cpu().numpy()
# def evaluate_and_plot(checkpoints, model, test_loader, device):
#     # model.eval()
#     y_true_list, y_pred_list, conf_list = [], [], []
#     with torch.no_grad():
#         for ckp in checkpoints:
#             model.load_state_dict(ckp)
#             model.eval()


#             for batch in test_loader:
#                 batch = batch.to(device)
#                 output = model(batch)

#                 mu = output["predict"]   # [B,1]
#                 nu = output["nu"]
#                 alpha = output["alpha"]
#                 beta = output["beta"]

#                 y_true = batch.label.view(-1, 1)  # 真实标签 [B,1]

#                 y_true_list.append(y_true.cpu().numpy())
#                 y_pred_list.append(mu.cpu().numpy())
#                 conf_list.append(compute_conf(mu, nu, alpha, beta))
    
#     # 拼接所有样本
#     y_true = np.concatenate(y_true_list, axis=0).squeeze()
#     y_pred = np.concatenate(y_pred_list, axis=0).squeeze()
#     conf = np.concatenate(conf_list, axis=0).squeeze()
#     # 归一化
#     conf = (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)

#     # 按置信度分桶
#     n_bins = 5
#     bin_edges = np.linspace(0, 1, n_bins+1)
#     rmse_vals, conf_centers = [], []

#     for i in range(n_bins):
#         mask = (conf >= bin_edges[i]) & (conf < bin_edges[i+1])
#         if mask.sum() > 0:
#             rmse = rmse(y_true[mask], y_pred[mask])
#             rmse_vals.append(rmse)
#             conf_centers.append((bin_edges[i] + bin_edges[i+1]) / 2)

#     # 绘制曲线
#     plt.figure(1,figsize=(6,4))
#     plt.plot(conf_centers, rmse_vals, marker="o", linestyle="-")
#     plt.xlabel("Confidence")
#     plt.ylabel("RMSE")
#     plt.title("RMSE vs Confidence on Test Set")
#     plt.grid(True)
#     plt.savefig('不确定性和RMSE相关图_idea.png', dpi=300, bbox_inches="tight")
#     plt.close(1)

#     # 保存成csv文件，这样方便导入graphpad
#     df = pd.DataFrame({
#         "Confidence": conf_centers,  # 第一列：置信度中心
#         "RMSE": rmse_vals            # 第二列：对应分桶的RMSE
#     })
#     df.to_csv("confidence_vs_rmse_idea.csv", index=False)  # index=False表示不保存行索引
#     print("CSV文件已保存为：confidence_vs_rmse.csv")
