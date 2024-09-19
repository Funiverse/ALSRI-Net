import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import  os
SMI_CHAR_SET = {'<MASK>': 0,'C': 1, ')': 2, '(': 3, 'c': 4, 'O': 5, ']': 6, '[': 7,
             '@': 8, '1': 9, '=': 10, 'H': 11, 'N': 12, '2': 13, 'n': 14,
             '3': 15, 'o': 16, '+': 17, '-': 18, 'S': 19, 'F': 20, 'p': 21,
             'l': 22, '/': 23, '4': 24, '#': 25, 'B': 26, '\\': 27, '5': 28,
             'r': 29, 's': 30, '6': 31, 'I': 32, '7': 33, '%': 34, '8': 35,
             'e': 36, 'P': 37, '9': 38, 'R': 39, 'u': 40, '0': 41, 'i': 42,
             '.': 43, 'A': 44, 't': 45, 'h': 46, 'V': 47, 'g': 48, 'b': 49,
             'Z': 50, 'T': 51, 'M': 52}

SEQ_CHAR_SET = {'<MASK>': 0,'A': 1, 'C': 2, 'D': 3, 'E': 4,
                'F': 5, 'G': 6, 'H': 7, 'K': 8,
                'I': 9, 'L': 10, 'M': 11, 'N': 12,
                'P': 13, 'Q': 14, 'R': 15, 'S': 16,
                'T': 17, 'V': 18, 'Y': 19, 'W': 20}



# Return an array with the same dimension as seq and all<MASK>except for position
def position_seq(seq, position):
    res = ['<MASK>']*len(seq)
    for i in position:
        res[i-1] = seq[i-1]
    return res
# Return an array with a length of max_ SMI_ Len, obtain the SMILES encoding format of the input sequence (1 * 256), use 0 as padding for sentence length filling
def label_smiles(line, smi_len):
    label = np.zeros(smi_len)
    for i, lab in enumerate(line[:smi_len]):
        label[i] = SMI_CHAR_SET[lab]
    return label
# Return an array with a length of max_ Seq_ Len, obtain the seq encoding format of the input sequence (1 * 1024)
def label_seq(line, seq_len):
    label = np.zeros(seq_len)
    for i, lab in enumerate(line[:seq_len]):
        label[i] = SEQ_CHAR_SET[lab]
    return label

class MyDataset(Dataset):
    def __init__(self, data_path,data_set_name,protein_seq_len,pocket_seq_len,smi_len): # file_list是文件对应的pdb_id
        self.data_path = data_path
        self.data_set =data_set_name
        self.protein_seq_len = protein_seq_len
        self.pocket_seq_len = pocket_seq_len
        self.smi_len = smi_len

        #affinity dict
        affinity = {}
        affinity_path = os.path.join(data_path,'affinity.csv')
        affinity_data = pd.read_csv(affinity_path)
        for _, row in affinity_data.iterrows():
            affinity[row.iloc[0]] = row.iloc[1]
        self.affinity = affinity

        #seq dict
        pdbid = {}
        smile = {}
        protein_seq = {}
        pocket_seq = {}
        position = {}
        seq_path = os.path.join(data_path,f'{data_set_name}_seq.csv')
        seq_data = pd.read_csv(seq_path)

        i = 0
        for _, row in seq_data.iterrows():
            pdbid[i] = row.iloc[1]
            smile[row.iloc[1]] = row.iloc[2]
            protein_seq[row.iloc[1]] = row.iloc[3]
            pocket_seq[row.iloc[1]] = row.iloc[4]
            position[row.iloc[1]] = eval(row.iloc[5])
            i += 1
        self.position = position
        self.smile = smile
        self.protein_seq = protein_seq
        self.pocket_seq = pocket_seq
        self.pdbid = pdbid
        self.len = len(self.protein_seq)
        assert len(protein_seq) == len(smile) == len(pocket_seq) , 'len(seq) != len(smile) ,please check your data!'

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        pdbid = self.pdbid[idx]


        smi = self.smile[pdbid]
        pro_seq = self.protein_seq[pdbid]
        poc_seq = self.pocket_seq[pdbid]
        pos = self.position[pdbid]
        mask_poc_seq = position_seq(pro_seq, pos)

        smi_encode = torch.tensor(label_smiles(smi, self.smi_len)).long()
        pro_seq_encode = torch.tensor(label_seq(pro_seq, self.protein_seq_len)).long()
        poc_seq_encode = torch.tensor(label_seq(poc_seq, self.pocket_seq_len)).long()
        mask_poc_encode = torch.tensor(label_seq(mask_poc_seq, self.protein_seq_len)).long()
        affinity = torch.tensor(np.array(self.affinity[pdbid], dtype=np.float32))

        return pdbid,smi_encode,pro_seq_encode,poc_seq_encode,mask_poc_encode,affinity