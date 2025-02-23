import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np


class HLApepDataset(Dataset):
    def __init__(self, HLA, peptide, use_mean):
        super(HLApepDataset, self).__init__()
        self.HLA = torch.tensor(np.array(HLA))
        self.peptide = peptide
        self.use_mean = use_mean

    def __getitem__(self, idx):
        if self.use_mean:
            self.HLA = torch.mean(self.HLA, dim=0)
            self.peptide = torch.mean(self.peptide, dim=0)
        return self.HLA[idx], self.peptide[idx]
    
    def __len__(self):
        return self.HLA.shape[0]
    


def generate_representation(seqvec, seq_list):
    '''seqvec: pre-trained model
       char_seq_list: pep seq list
    '''
    embeddings_list = []
    for seq in seq_list:
        embedding  = seqvec.embed_sentence(list(seq))
        residue_embd = torch.tensor(embedding).sum(dim=0)
        L, _ = residue_embd.shape
        if L < 34:
            padding = (0, 0, 0, 34 - L)
            residue_embd = F.pad(residue_embd, padding)
        embeddings_list.append(residue_embd)
    final_tensor = torch.stack(embeddings_list)
    return final_tensor
