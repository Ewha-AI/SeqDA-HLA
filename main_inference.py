import os, argparse, random, h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from allennlp.commands.elmo import ElmoEmbedder
from utils.data import HLApepDataset, generate_representation
from utils.mhc_total import mhc_pseudo
from model.seqdahla import SeqDAHLA

import torch
from torch.utils.data import DataLoader
import torch.autograd as autograd
autograd.set_detect_anomaly(True)
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def test_model(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        for hla_emb, pep_emb in iter(test_loader):
            hla_emb = hla_emb.to(device) 
            pep_emb = pep_emb.to(device)
            pep_emb = pep_emb.squeeze(dim=1)
            output, attn = model(hla_emb, pep_emb)
            model_prob = torch.sigmoid(output).to('cpu').numpy()
    return model_prob, attn


def main(peptides, hlas, bn):
    # Train/Test settings
    parser = argparse.ArgumentParser(description='SeqDA-HLA')
    parser.add_argument('--model', type=str, default='HLA_seedout', help='model name')
    parser.add_argument('--opt', type=str, default='Mhc-Pep_', help='model option')
    parser.add_argument('--n_workers', type=int, default=0, help='number of workers')
    parser.add_argument('--thsd', type=float, default=0.5, help='threshold')
    parser.add_argument('--seed', type=int, default=42, help='seed for fixing configuration')
    parser.add_argument('--start_lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--end_lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--wd', type=float, default=0, help='weight decay value (L2 penalty): 0/1e-6') 
    parser.add_argument('--batch', type=int, default=512, help='batch size')
    parser.add_argument('--test_batch', type=int, default=1, help='test batch size')
    parser.add_argument('--epoch', type=int, default=50, help='epoch')
    parser.add_argument('--embedding-dim', type=int, default=1024, help='dimension of embedding')
    parser.add_argument('--n_heads', type=int, default=3, help='Number of head attentions')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type')
    parser.add_argument('--resume-from', type=int, default=None, help='Epoch to resume from')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--d_model', type=int, default=64, help='dim of attn weight matrices')
    parser.add_argument('--d_qkv', type=int, default=64, help='dim of q,k,v')
    parser.add_argument('--d_ff', type=int, default=64, help='dim for feedforward layers')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='dropout rate of the classifier')
    args = parser.parse_args()

    CFG = {
        'NUM_WORKERS':args.n_workers,
        'EPOCHS':args.epoch,
        'START_LEARNING_RATE':args.start_lr,
        'END_LEARNING_RATE':args.end_lr,
        'WEIGHT_DECAY':args.wd,
        'BATCH_SIZE':args.batch,
        'Test_BATCH_SIZE': bn,
        'THRESHOLD':args.thsd,
        'SEED':args.seed,
        'embedding_dim': args.embedding_dim,
        'n_heads': args.n_heads,
        'optimizer': args.optimizer,
        'd_model':args.d_model,
        'd_qkv':args.d_qkv
    }
    seed_everything(CFG['SEED'])

    # Elmo embedding---SeqVec
    # weights and options files can be downloaded from https://github.com/rostlab/SeqVec
    weights = '../../../weights.hdf5'
    options = '../../../options.json'
    # Stabilize internal state by dummy sequence
    dummy_peptide = ['YTDQFSRNY','AVAPFFKSY','LLYESPERY','LSDLGRLSY','QIFNKPYWL'] 

    pep_emb_list = generate_representation(weights, options, dummy_peptide, peptides)
        
    hla_emb_list = []
    with h5py.File('utils/total_HLA_embeddings.hdf5', 'r') as f:
        for hla in hlas:
            hla_emb_list.append(np.array(f[hla]))
    
    test_dataset = HLApepDataset(hla_emb_list, pep_emb_list, use_mean=False)
    test_loader = DataLoader(test_dataset, batch_size=bn, shuffle=False, num_workers=CFG['NUM_WORKERS'])

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = SeqDAHLA(args)
    model_checkpoint = torch.load('model/model.pth')
    model_checkpoint = {k: v for k, v in model_checkpoint.items() if k in model_checkpoint}
    model.load_state_dict(model_checkpoint['model_state_dict'])
    model = model.to(device)
    
    prob_test, attn = test_model(model, test_loader, device)
    prob_test = np.atleast_1d(prob_test)
    final_pred = [1 if i>=0.5 else 0 for i in prob_test]

    return prob_test, final_pred, attn


def draw_attn(hlas, peps, attn, attn_type, cs):
    n_heads = 3
    attn_type_dict = {'aqq': 0, 'aqs': 1, 'cross_a': 2, 'self_a': 3}

    attns = attn[attn_type_dict[attn_type]]  # [bs, heads, 34, 34]

    for start_idx in range(0, len(hlas), cs):
        end_idx = min(start_idx + cs, len(hlas))
        fig, axes = plt.subplots(nrows=(end_idx - start_idx), ncols=1, figsize=(5, (end_idx - start_idx) * 2))

        if end_idx - start_idx == 1:
            axes = [axes]

        for i, (hla, pep) in enumerate(zip(hlas[start_idx:end_idx], peps[start_idx:end_idx])):
            for head in range(n_heads):
                if head == 0:
                    temp = np.array(attns[start_idx + i, head, :, :len(pep)].cpu())
                else:
                    temp += np.array(attns[start_idx + i, head, :, :len(pep)].cpu())

            hla_pseudo = mhc_pseudo[hla]
            temp_pd = pd.DataFrame(np.array(temp), index=list(hla_pseudo), columns=list(pep)).T

            fig.patch.set_facecolor('white')
            cmap = 'cividis'
            sns.heatmap(temp_pd, ax=axes[i], cmap=cmap, square=True, cbar=False)

            axes[i].set_title('{} | {} | len{} '.format(hla, pep, len(pep)), fontsize=13)
            axes[i].set_xlabel('HLA', fontsize=13)
            axes[i].set_ylabel('Peptide', fontsize=13)

            axes[i].set_yticks(range(len(pep)))
            axes[i].set_yticklabels(list(pep), fontsize=11, rotation=0)

            axes[i].set_xticks(range(len(hla_pseudo)))
            axes[i].set_xticklabels(list(hla_pseudo), fontsize=11)

        plt.tight_layout()
        hla_name = hla.replace('*', '').replace(':', '')
        plt.savefig('./out_{}.jpg'.format(hla_name), bbox_inches='tight', dpi=600)
        plt.close(fig)


if __name__ == '__main__':

    hlas = ["HLA-A*01:01", "HLA-A*02:01", "HLA-B*18:01"]
    peptides = ['YTDQFSRNY','AVAPFFKSY','LLYESPERY','LSDLGRLSY']  # Make sure there are no duplicate peptide sequences

    hla_in = [hla for hla in hlas for _ in range(len(peptides))]
    peptide_in = peptides * len(hlas)
    
    probability, prediction, attn = main(peptide_in, hla_in, len(hla_in))

    # All possible HLA-peptide pairs
    print("HLA \t\t Peptide \t Probability \t Binding")
    for i, j, k, l in zip(hla_in, peptide_in, probability, prediction):
        print("%s \t %s \t" % (i, j), "{:.8f}".format(k), "\t %d" % l)
    
    draw_attn(hla_in, peptide_in, attn, 'self_a', len(peptides))
