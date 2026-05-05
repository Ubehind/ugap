from argparse import Namespace

import torch.nn as nn
import torch
import torch.nn.functional as F
from .mpn import MPN
from chemprop.nn_utils import get_activation_function, initialize_weights


class MoleculeModel(nn.Module):
    """A MoleculeModel is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, classification: bool, multiclass: bool):
        """
        Initializes the MoleculeModel.

        :param classification: Whether the model is a classification model.
        """
        super(MoleculeModel, self).__init__()

        self.classification = classification
        if self.classification:
            self.sigmoid = nn.Sigmoid()
        self.multiclass = multiclass
        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)
        assert not (self.classification and self.multiclass)

    def create_encoder(self, args: Namespace):
        """
        Creates the message passing encoder for the model.

        :param args: Arguments.
        """
        self.encoder = MPN(args)

    def create_ffn(self, args: Namespace):
        """
        Creates the feed-forward network for the model.

        :param args: Arguments.
        """
        self.multiclass = args.dataset_type == 'multiclass'
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            first_linear_dim = args.hidden_size * 1
            if args.use_input_features:
                first_linear_dim += args.features_dim

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)


        self.num_tasks = args.output_size
        if args.dataset_type == 'classification':
            if args.ffn_num_layers == 1:
                ffn = [
                    dropout,
                    nn.Linear(first_linear_dim, args.output_size * 2)
                ]
            else:
                ffn = [
                    dropout,
                    nn.Linear(first_linear_dim, args.ffn_hidden_size)
                ]
                for _ in range(args.ffn_num_layers - 2):
                    ffn.extend([
                        activation,
                        dropout,
                        nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                    ])
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args.ffn_hidden_size, args.output_size * 2),
                ])
            self.ffn = nn.Sequential(*ffn)
        else:

            ffn_shared = [dropout]
            if args.ffn_num_layers > 1:
                ffn_shared.append(nn.Linear(first_linear_dim, args.ffn_hidden_size))
                for _ in range(args.ffn_num_layers - 2):
                    ffn_shared.extend([
                        activation,
                        dropout,
                        nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                    ])
                ffn_shared.extend([activation, dropout])
                last_hidden_dim = args.ffn_hidden_size
            else:
                last_hidden_dim = first_linear_dim
            self.ffn = nn.Sequential(*ffn_shared)
            self.mu_layer = nn.Linear(last_hidden_dim, 1)      
            self.evidence_layer = nn.Linear(last_hidden_dim, 3)
            nn.init.constant_(self.evidence_layer.bias[0], 1.0) 
            nn.init.constant_(self.evidence_layer.bias[1], 2.0)
        

    def forward(self, *input):
        """
        Runs the MoleculeModel on input.

        :param input: Input.
        :return: The output of the MoleculeModel.
        """
        output = self.encoder(*input)
        if self.classification:
            logits = self.ffn(output['mol_vecs']).reshape((output['mol_vecs'].shape[0], self.num_tasks, 2))
            alphas = nn.functional.softplus(logits) + 1
            S = torch.sum(alphas, dim=-1, keepdim=True)
            output['alphas'] = alphas
            output['preds'] = alphas / S
            output['logits'] = logits
            
        else:
            # logits = self.ffn(output['mol_vecs'])
            shared_out = self.ffn(output['mol_vecs'])
            mu = self.mu_layer(shared_out)
            evid_params = self.evidence_layer(shared_out)
            nu = F.softplus(evid_params[:, 0:1]) + 0.01
            alpha = F.softplus(evid_params[:, 1:2]) + 1.0 + 0.01
            beta = F.softplus(evid_params[:, 2:3]) + 0.01
            output['mu'] = mu
            output['nu'] = nu
            output['alpha'] = alpha
            output['beta'] = beta
            output['preds'] = mu
        return output
    
    def forward_with_emb(self, *input):
        output = self.encoder(*input)
        if self.classification:
            logits = self.ffn(output['mol_vecs']).reshape((output['mol_vecs'].shape[0], self.num_tasks, 2))
            alphas = nn.functional.softplus(logits) + 1
            S = torch.sum(alphas, dim=-1, keepdim=True)
            output['alphas'] = alphas
            output['preds'] = alphas / S
            output['logits'] = logits
        else:
            shared_out = self.ffn(output['mol_vecs'])
            mu = self.mu_layer(shared_out)
            evid_params = self.evidence_layer(shared_out)
            nu = F.softplus(evid_params[:, 0:1]) + 0.01
            alpha = F.softplus(evid_params[:, 1:2]) + 1.0 + 0.01
            beta = F.softplus(evid_params[:, 2:3]) + 0.01




            output['mu'] = mu
            output['nu'] = nu
            output['alpha'] = alpha
            output['beta'] = beta
            output['preds'] = mu
        return output


def build_model(args: Namespace) -> nn.Module:
    """
    Builds a MoleculeModel, which is a message passing neural network + feed-forward layers.

    :param args: Arguments.
    :return: A MoleculeModel containing the MPN encoder along with final linear layers with parameters initialized.
    """
    output_size = args.num_tasks
    args.output_size = output_size
    if args.dataset_type == 'multiclass':
        args.output_size *= args.multiclass_num_classes

    model = MoleculeModel(classification=args.dataset_type == 'classification', multiclass=args.dataset_type == 'multiclass')
    model.create_encoder(args)
    model.create_ffn(args)

    initialize_weights(model)

    return model
