import os
from sklearn.metrics import mean_squared_error, roc_auc_score,average_precision_score
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns
matplotlib.use('Agg')



def save_confidence_vs_rmse(y_true, y_pred, confidences, n_bins=5, save_path='images/conf_vs_rmse.png'):
 
    dir_name = os.path.dirname(save_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)


    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    conf = np.array(confidences).flatten()
    

    sorted_indices = np.argsort(conf)
    y_true_sorted = y_true[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]
    conf_sorted = conf[sorted_indices]
    
    n_samples = len(conf)
    samples_per_bin = n_samples // n_bins
    
    bin_rmses = []
    bin_conf_means = [] 
    bin_labels = []     

    for i in range(n_bins):

        start_idx = i * samples_per_bin

        end_idx = (i + 1) * samples_per_bin if i != n_bins - 1 else n_samples
        

        b_true = y_true_sorted[start_idx:end_idx]
        b_pred = y_pred_sorted[start_idx:end_idx]
        b_conf = conf_sorted[start_idx:end_idx]
        

        rmse = np.sqrt(mean_squared_error(b_true, b_pred))
        bin_rmses.append(rmse)
        bin_conf_means.append(np.mean(b_conf))

        label = f"{b_conf.min():.2f}-{b_conf.max():.2f}"
        bin_labels.append(label)

    plt.figure(figsize=(10, 6))

    x_axis = np.arange(1, n_bins + 1)
    plt.plot(x_axis, bin_rmses, marker='s', markersize=8, linestyle='-', 
             linewidth=2, color='#d62728', label='RMSE per Quantile')

    for x, y in zip(x_axis, bin_rmses):
        plt.text(x, y + (max(bin_rmses)*0.02), f'{y:.3f}', ha='center', va='bottom', fontsize=10)
    plt.xticks(x_axis, bin_labels, rotation=15)
    plt.xlabel('Confidence Range (Quantile Bins)', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.title(f'Quantile-based Confidence vs. RMSE (n_bins={n_bins})', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)
    

    plt.annotate(f'Samples per bin: ~{samples_per_bin}', xy=(0.05, 0.95), 
                 xycoords='axes fraction', fontsize=10, bbox=dict(boxstyle="round", fc="w"))


    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Quantile plot saved to: {save_path}")

    return bin_labels, bin_rmses

# rmse cutoff
def plot_cutoff(y_true, y_pred, confidences, save_path='images/cutoff_analysis.png',csv_file_name="cutoffs.csv"):

    dir_name = os.path.dirname(save_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)

    y_true = np.array(y_true).astype(float).flatten()
    y_pred = np.array(y_pred).astype(float).flatten()
    conf = np.array(confidences).astype(float).flatten()

    if len(y_true) == 0:
        print("Error: Input data is empty.")
        return


    indices = np.argsort(conf)[::-1]
    

    y_true_sorted = y_true[indices]
    y_pred_sorted = y_pred[indices]
    

    fractions = np.linspace(0.1, 1.0, 20)
    rmses = []
    
    for f in fractions:
        num_samples = int(len(y_true_sorted) * f)
        if num_samples < 1:
            num_samples = 1 
            
        f_true = y_true_sorted[:num_samples]
        f_pred = y_pred_sorted[:num_samples]
        mse = mean_squared_error(f_true, f_pred)
        rmses.append(np.sqrt(mse))
    
    export_df = pd.DataFrame({
        'Remaining_Fraction_Percent': fractions * 100,
        'RMSE': rmses
    })
    export_df.to_csv(csv_file_name, index=False)
    print(f"Cutoff data for GraphPad successfully saved to: {csv_file_name}")

    plt.figure(figsize=(8, 5))

    plt.plot(fractions * 100, rmses, marker='D', markersize=6, 
             linestyle='-', color='#2ca02c', linewidth=2, label='Remaining RMSE')

    plt.xlabel('Remaining Data Fraction (%) - [High Conf $\leftarrow$ Total]', fontsize=12)
    plt.ylabel('RMSE (Lower is Better)', fontsize=12)
    plt.gca().invert_xaxis() 
    
    plt.title('Sparsification Curve: Error Dropdown with Confidence Cutoffs', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Cutoff experiment plot successfully saved to: {save_path}")

    return fractions, rmses

def plot_classification_experiments(probs, labels, conf, save_dir='images'):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    flat_probs = np.array(probs).astype(float).flatten()
    flat_labels = np.array(labels).astype(float).flatten()
    flat_conf = np.array(conf).astype(float).flatten()

    valid_mask = ~np.isnan(flat_labels) & (flat_labels != -1)
    flat_probs = flat_probs[valid_mask]
    flat_labels = flat_labels[valid_mask]
    flat_conf = flat_conf[valid_mask]


    fractions = np.linspace(0.1, 1.0, 10)
    aucs = []
    indices = np.argsort(flat_conf)[::-1]
    
    for f in fractions:
        num = int(len(indices) * f)
        idx = indices[:num]

        if len(np.unique(flat_labels[idx])) < 2:
            aucs.append(aucs[-1] if aucs else 0.5)
            continue
        aucs.append(roc_auc_score(flat_labels[idx], flat_probs[idx]))
        f_true = flat_labels[idx]
        num_pos = np.sum(f_true == 1)
        num_neg = np.sum(f_true == 0)
        print(f"Fraction {f:.1f}: Samples={len(f_true)}, Pos={num_pos}, Neg={num_neg}, AUC={roc_auc_score(flat_labels[idx], flat_probs[idx]):.3f}")

    plt.figure(figsize=(8, 5))
    plt.plot(fractions * 100, aucs, marker='o', color='red')
    plt.gca().invert_xaxis()
    plt.xlabel('Remaining Data % (High Conf $\leftarrow$ Total)')
    plt.ylabel('Average AUC-ROC')
    plt.title('Classification Sparsification Curve')
    plt.savefig(f'{save_dir}/class_cutoff_auc.png')
    plt.close()

    bins = np.linspace(0, 1, 11)
    bin_accs = []
    bin_confs = []
    for i in range(10):
        mask = (flat_probs >= bins[i]) & (flat_probs < bins[i+1])
        if np.any(mask):
            acc = np.mean((flat_probs[mask] > 0.5) == flat_labels[mask])
            bin_accs.append(acc)
            bin_confs.append(np.mean(flat_probs[mask]))
    
    plt.figure(figsize=(6, 6))
    plt.bar(bin_confs, bin_accs, width=0.1, alpha=0.7, edgecolor='black')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Observed Accuracy')
    plt.title('Reliability Diagram (ECE)')
    plt.savefig(f'{save_dir}/reliability_diagram.png')
    plt.close()

    print(f"Classification plots saved to {save_dir}")


def save_quantile_conf_vs_accuracy(y_true, y_probs, confidences, n_bins=5, save_path='images/class_conf_vs_acc.png', filepath="classification.csv"):

    dir_name = os.path.dirname(save_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)

    y_true = np.array(y_true).astype(float).flatten()
    y_probs = np.array(y_probs).astype(float).flatten()
    conf = np.array(confidences).astype(float).flatten()

    valid_mask = ~np.isnan(y_true) & (y_true != -1)
    y_true = y_true[valid_mask]
    y_probs = y_probs[valid_mask]
    conf = conf[valid_mask]

    if len(y_true) == 0:
        print("Error: No valid samples found for plotting.")
        return

    sorted_indices = np.argsort(conf)
    y_true_sorted = y_true[sorted_indices]
    y_probs_sorted = y_probs[sorted_indices]
    conf_sorted = conf[sorted_indices]

    n_samples = len(conf_sorted)
    samples_per_bin = n_samples // n_bins
    
    bin_accs = []
    bin_conf_labels = []
    bin_conf_min = []
    bin_conf_max = []

    y_hard_pred = (y_probs_sorted > 0.5).astype(float)

    for i in range(n_bins):
        start_idx = i * samples_per_bin
        end_idx = (i + 1) * samples_per_bin if i != n_bins - 1 else n_samples

        b_true = y_true_sorted[start_idx:end_idx]
        b_pred = y_hard_pred[start_idx:end_idx]
        b_conf = conf_sorted[start_idx:end_idx]

        acc = np.mean(b_true == b_pred)
        bin_accs.append(acc)
        

        label = f"{b_conf.min():.2f}-{b_conf.max():.2f}"
        bin_conf_labels.append(label)

        bin_conf_min.append(b_conf.min())
        bin_conf_max.append(b_conf.max())
    
    
    
    df_export = pd.DataFrame({
        'Bin_Index': np.arange(1, n_bins + 1),
        'Confidence_Range': bin_conf_labels,
        'Conf_Min': bin_conf_min,
        'Conf_Max': bin_conf_max,
        'Accuracy': bin_accs
    })
    df_export.to_csv(filepath, index=False)
    print(f"Classification calibration data saved to: {filepath}")

    plt.figure(figsize=(10, 6))
    x_axis = np.arange(1, n_bins + 1)

    bars = plt.bar(x_axis, bin_accs, color='#17becf', alpha=0.7, edgecolor='black', width=0.6)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random Guess (0.5)')

    plt.xticks(x_axis, bin_conf_labels, rotation=15)
    plt.xlabel('Confidence Range (Quantile Bins)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Classification: Confidence Quantiles vs. Accuracy', fontsize=14)
    plt.ylim(0, 1.1) 
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.legend()


    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Classification Quantile plot saved to: {save_path}")

    return bin_conf_labels, bin_accs





def plot_binned_boxplot(final_deltas, final_s_clean, num_bins=3,filename='graphpad.csv'):

    df = pd.DataFrame({
        'Delta': final_deltas,
        'S_clean': final_s_clean
    })

    bin_labels = ['Low Δ', 'Mid Δ', 'High Δ', 'Extreme Δ'][:num_bins]
    df['Delta_Bin'] = pd.qcut(df['Delta'], q=num_bins, labels=bin_labels)

    wide_df = pd.DataFrame()
    for label in bin_labels:

        group_data = df[df['Delta_Bin'] == label]['S_clean'].reset_index(drop=True)
        wide_df[label] = group_data

    wide_df.to_csv(filename, index=False)
    print(f"export to: {filename}")
 
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("Blues_r", n_colors=num_bins)
    ax = sns.boxplot(x='Delta_Bin', y='S_clean', data=df, 
                     palette=palette, showfliers=False, width=0.6)

    sns.stripplot(x='Delta_Bin', y='S_clean', data=df.sample(min(len(df), 500)), 
                  size=2, color=".3", linewidth=0, alpha=0.3, ax=ax)

    plt.title('Evidence Distribution across Stability Levels', fontsize=14)
    plt.xlabel('Predictive Discrepancy (Δ) Bins')
    plt.ylabel('Evidence Magnitude (S_clean)')
    plt.tight_layout()
    plt.savefig('binned_boxplot.png', dpi=300)


def save_quantile_conf_vs_auc(y_true, y_probs, confidences, n_bins=5, save_path='images/class_conf_vs_auc.png', filepath="classification_auc.csv"):

    dir_name = os.path.dirname(save_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)

    y_true = np.array(y_true).astype(float).flatten()
    y_probs = np.array(y_probs).astype(float).flatten()
    conf = np.array(confidences).astype(float).flatten()

    valid_mask = ~np.isnan(y_true) & (y_true != -1)
    y_true = y_true[valid_mask]
    y_probs = y_probs[valid_mask]
    conf = conf[valid_mask]

    if len(y_true) == 0:
        print("Error: No valid samples found for plotting.")
        return

    sorted_indices = np.argsort(conf)
    y_true_sorted = y_true[sorted_indices]
    y_probs_sorted = y_probs[sorted_indices]
    conf_sorted = conf[sorted_indices]

    n_samples = len(conf_sorted)
    samples_per_bin = n_samples // n_bins
    
    bin_aucrocs = []
    bin_aucprs = []
    bin_conf_labels = []
    bin_conf_min = []
    bin_conf_max = []

    for i in range(n_bins):
        start_idx = i * samples_per_bin
        end_idx = (i + 1) * samples_per_bin if i != n_bins - 1 else n_samples

        b_true = y_true_sorted[start_idx:end_idx]
        b_prob = y_probs_sorted[start_idx:end_idx]
        b_conf = conf_sorted[start_idx:end_idx]

        try:
            aucroc = roc_auc_score(b_true, b_prob)
            aucpr = average_precision_score(b_true, b_prob)
        except ValueError:

            aucroc = 0.5
            aucpr = np.mean(b_true) 
        
        bin_aucrocs.append(aucroc)
        bin_aucprs.append(aucpr)
        

        label = f"{b_conf.min():.2f}-{b_conf.max():.2f}"
        bin_conf_labels.append(label)
        bin_conf_min.append(b_conf.min())
        bin_conf_max.append(b_conf.max())

    df_export = pd.DataFrame({
        'Bin_Index': np.arange(1, n_bins + 1),
        'Confidence_Range': bin_conf_labels,
        'Conf_Min': bin_conf_min,
        'Conf_Max': bin_conf_max,
        'AUC_ROC': bin_aucrocs,
        'AUC_PR': bin_aucprs
    })
    df_export.to_csv(filepath, index=False)
    print(f"Classification metrics saved to: {filepath}")

    plt.figure(figsize=(10, 6))
    x_axis = np.arange(1, n_bins + 1)

    bars = plt.bar(x_axis, bin_aucrocs, color='#ff7f0e', alpha=0.7, edgecolor='black', width=0.6)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random (0.5)')

    plt.xticks(x_axis, bin_conf_labels, rotation=15)
    plt.xlabel('Confidence Range (Quantile Bins)', fontsize=12)
    plt.ylabel('AUC-ROC', fontsize=12)
    plt.title('Classification: Confidence vs. Discriminative Power (AUC)', fontsize=14)
    plt.ylim(0, 1.1) 
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.legend()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return bin_conf_labels, bin_aucrocs




from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import Draw




def visualize_mol_perturbation(smiles, batch, mol_id, local_node_indices, save_path, preds, target, u=None):

    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        return
    
    if isinstance(target, (float, int)) or target.shape == (): 

        p_val = preds[1].item() if hasattr(preds, 'item') else preds[1]
        t_val = int(target)
        legend_text = f"Pred: {p_val:.3f} | Target: {t_val} | u: {u:.3f}   " + smiles
    else: 
        p_val = float(preds.flatten()[0])
        t_val = float(target.flatten()[0])
        legend_text = f"Pred: {p_val:.3f} | Target: {t_val:.3f} " + smiles

    Chem.rdDepictor.Compute2DCoords(mol)
    mol = rdMolDraw2D.PrepareMolForDrawing(mol)

    highlight_atoms = [int(i) for i in local_node_indices]
    highlight_colors = {i: (1.0, 0.7, 0.7) for i in highlight_atoms} 

    drawer = rdMolDraw2D.MolDraw2DSVG(400, 450) 
    save_path = save_path.replace('.pdf', '.svg')
        
    options = drawer.drawOptions()
    options.addStereoAnnotation = True
    options.bondLineWidth = 2 
    options.padding = 0.18
    options.highlightRadius = 0.35 
    options.prepareMolsBeforeDrawing = True
    

    options.legendFontSize = 25 

    drawer.DrawMolecule(
        mol, 
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_colors,
        legend=legend_text 
    )
    drawer.FinishDrawing()

    text = drawer.GetDrawingText()
    mode = 'wb' if isinstance(text, bytes) else 'w'
    
    with open(save_path, mode) as f:
        f.write(text)
        
    print(f"Successfully saved visualization with values to: {save_path}")




    


def analysis_calibration_ece(preds, confs, targets, n_bins=10, title="Reliability Diagram"):


    preds = np.array(preds).flatten()
    confs = np.array(confs).flatten()
    targets = np.array(targets).flatten()

    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_boundaries = np.unique(np.percentile(confs, quantiles))
    actual_bins = len(bin_boundaries) - 1
    
    bin_accs = []
    bin_confs = []
    bin_sizes = []
    ece = 0.0
    true_ece = 0.0


    for i in range(actual_bins):
        if i == actual_bins - 1:
            indices = np.where((confs >= bin_boundaries[i]) & (confs <= bin_boundaries[i+1]))[0]
        else:
            indices = np.where((confs >= bin_boundaries[i]) & (confs < bin_boundaries[i+1]))[0]
        
        if len(indices) > 0:
            bin_preds = (preds[indices] > 0.5).astype(int)
            acc = np.mean(bin_preds == targets[indices])
            avg_conf = np.mean(confs[indices])
            
            bin_accs.append(acc)
            bin_confs.append(avg_conf)
            bin_sizes.append(len(indices))
            
            ece += np.abs(acc - avg_conf) * (len(indices) / len(confs))
            true_ece += (avg_conf - acc) * (len(indices) / len(confs))

    plt.figure(figsize=(7, 7), dpi=100)

    plt.plot(bin_confs, bin_accs, marker='o', color="#e34a33", linewidth=2, 
             label="Accuracy Line", zorder=3)
    

    plt.bar(bin_confs, bin_confs, width=0.05, color="#43a2ca", alpha=0.5, 
            edgecolor="#0868ac", label="Confidence (Bin Avg)", zorder=2)
    

    for i in range(len(bin_accs)):
        plt.plot([bin_confs[i], bin_confs[i]], [bin_confs[i], bin_accs[i]], 
                 color="#e34a33", linewidth=1.5, linestyle=":", zorder=4)
    

    plt.text(0.05, 0.92, f"ECE: {ece:.4f}\nBias: {true_ece:.4f}", fontsize=12, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='#cccccc'))
    
    plt.xlabel("Confidence (Average)", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.title(title, fontsize=15, pad=15)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc="upper left")
    plt.grid(axis='both', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"{title}.png", dpi=300)
    plt.close()
    
    print("trueECE", true_ece)
    print("ECE", ece)
    return ece




def analysis_regression_calibration(y_true, y_pred, uncertainties, n_bins=10):

    y_true = np.array(y_true).flatten()
    ypred = np.array(y_pred).flatten()
    uncertainties = np.array(uncertainties).flatten()


    errors = (y_true - y_pred)**2

    idx = np.argsort(uncertainties)
    uncertainties = uncertainties[idx]
    errors = errors[idx]

    bin_size = len(errors) // n_bins
    bin_mses = []
    bin_vars = []
    
    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < n_bins - 1 else len(errors)
        
        bin_mses.append(np.mean(errors[start:end]))
        bin_vars.append(np.mean(uncertainties[start:end]))
    plt.figure(figsize=(6, 6))
    max_val = max(max(bin_mses), max(bin_vars))
    plt.plot([0, max_val], [0, max_val], '--', color='gray', label='Perfect Calibration')
    plt.scatter(bin_vars, bin_mses, color='#43a2ca', s=100, edgecolors='k', zorder=3)
    plt.fill_between(bin_vars, bin_mses, bin_vars, color='#e34a33', alpha=0.2, label='Calibration Gap')
    
    plt.xlabel('Predicted Variance (Uncertainty)', fontsize=12)
    plt.ylabel('Observed MSE (Error)', fontsize=12)
    plt.title('Regression Calibration Analysis (UCE)', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('regression_beta.png', bbox_inches='tight')


from scipy.stats import norm
def plot_calibration_curve(preds, y_true, total_vars, save_path='calibration_curve.png', beta_val=1):

    preds = np.array(preds).flatten()
    y_true = np.array(y_true).flatten()
    total_vars = np.array(total_vars).flatten()

    stds = np.sqrt(total_vars)

    expected_p = np.linspace(0, 1, 20)
    observed_p = []
    
    for p in expected_p:
        if p == 0:
            observed_p.append(0)
            continue

        z_score = norm.ppf(1 - (1 - p) / 2)
        

        lower_bound = preds - z_score * stds
        upper_bound = preds + z_score * stds
        

        in_interval = np.logical_and(y_true >= lower_bound, y_true <= upper_bound)
        observed_p.append(np.mean(in_interval))

    area_error = np.mean(np.abs(expected_p - np.array(observed_p)))

    suffix = f"beta_{beta_val}" if beta_val is not None else "default"
    data_save_path = save_path.replace('.png', f'_{suffix}.npz')
    np.savez(data_save_path, 
             expected_p=expected_p, 
             observed_p=observed_p, 
             area_error=area_error, 
             beta=beta_val)
    print(f"Calibration data saved to {data_save_path}")





    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration') 
    plt.plot(expected_p, observed_p, 'r-', label=f'Evidential (Error: {area_error:.3f})')
    
    plt.xlabel('Expected Confidence Level')
    plt.ylabel('Observed Confidence Level')
    plt.title('Uncertainty Calibration Curve (Figure 4c)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    

    plt.savefig(save_path)
    print(f"Calibration curve saved to {save_path}")
    plt.close()

