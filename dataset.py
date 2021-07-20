from typing_extensions import final
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import collections

from tqdm import tqdm


class Nucleotides:
  def __init__(self, seqs):
    self.nuc_pairs= self.make_pairs(seqs)
    print("total words in vocab", self.nuc_pairs)
    self.encoding= {w:i for i,w in enumerate(self.nuc_pairs,1)}
    self.decoding= {i:w for i,w in enumerate(self.nuc_pairs,1)}

  
  def make_pairs(self, seqs, clip=1):
    nuc_pairs= collections.Counter()

    for seq in tqdm(seqs):
      nuc_pairs.update(seq)

    # check why this statement is so important (84, remains same without it)
    for nucs in list(nuc_pairs.keys()):
      if nuc_pairs[nucs] < clip:
        nuc_pairs.pop(nucs)

    return list(sorted(nuc_pairs.keys()))


  def size(self):
    assert len(self.encoding) == len(self.decoding)
    return len(self.encoding)

  


class Sequences(Dataset):
    def __init__(self, seq_len=131):
      self.seq_len= seq_len
      print(self.seq_len)
      self.seqs= self.convert_seqs_to_words(self.load_data())
      self.seqs= self.get_size_specific_seqs(self.seqs)
      self.nucleotides= Nucleotides(self.seqs)
    

    def read_fasta(self,fp):
        name, seq = None, []
        for line in fp:
            line = line.rstrip()
            if line.startswith(">"):
                if name: yield (name, ''.join(seq))
                name, seq = line, []
            else:
                seq.append(line)
        if name: yield (name, ''.join(seq))
    
    def load_data(self):
      sequences = []
      # Reading FASTA fil
      with open("/raid/ahtisham/permissive_enhancers","r") as fp:
        for name, seq in self.read_fasta(fp):
          sequences.append(seq)
      print("Sequences Read Succesfully !!!!")
      print("Total Raw Sequences: ",len(sequences))
      return sequences
    
    def add_padding(self,seq, p_len):
      seq = seq + ("P" * p_len)
      return seq
  
  
    def convert_seqs_to_words(self,sequences):
      f_sequences = []
      for i in range(len(sequences)):
        temp = ""
        str_ = sequences[i]
        j=0
        if len(str_)%3!=0:
          n = len(str_)
          while n % 3 != 0:
            n+=1
            str_= self.add_padding(str_,n-len(str_))
        for k in range(0,len(str_)):
          j+=1
          if  j%3==0:
            temp = temp + str_[k-2:j] + ' ' 
          #j+=3
        f_sequences.append(temp) 
      f_sequences= [j.split() for j in f_sequences]
      return f_sequences
    
    def encode(self, seq):
      enc= self.nucleotides.encoding
      a = np.array([enc.get(c) for c in seq])
      return a
    
    
    def get_size_specific_seqs(self, seqs):
      final_sequences=[]
      for i in range(len(seqs)):
        if (len(seqs[i]) == 131):
          final_sequences.append(seqs[i])
        elif (len(seqs[i])>131):
            new_seqs=self.break_seq_into_smaller_chunks(seqs[i])
            for i in range(len(new_seqs)):
                final_sequences.append(new_seqs[i])
      print("Now the total seqs are:", len(final_sequences))
      
      # for i in range(len(final_sequences)):
      #     if len(final_sequences[i])>131:
      #         print(len(final_sequences[i]))
      return final_sequences
    
    def break_seq_into_smaller_chunks(self,seq):
        new_seqs=[]
        num_new_seqs = int(len(seq)/131)
        start_index=0
        final_index=131
        if num_new_seqs >=2:
            for i in range(num_new_seqs):
                new_seqs.append(seq[start_index:final_index])
                start_index+=131
                final_index+=131
        elif(num_new_seqs<2):
            new_seqs.append(seq[0:final_index])
                    
        return new_seqs
                
                
    
    def __len__(self):
      return len(self.seqs)

    def __getitem__(self,i):
      return torch.from_numpy(self.encode(self.seqs[i]))



def load(batch_size, seq_len):
  data= Sequences(seq_len)
  return (DataLoader(data, batch_size, shuffle=True), data.nucleotides)
