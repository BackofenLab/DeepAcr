import os.path as osp
import torch
import glob
import random
import sys
import numpy as np
import torch
from Bio.Data import IUPACData
from Bio.SeqUtils import ProtParamData # Dipeptide Instability Weight Values (DIWV)
                                       # and Kyte and Doolittle's hydrophobicity scale.

from collections import OrderedDict
from itertools import product
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio import SeqIO
import yaml
from torch_geometric.data import Data
import psutil
import subprocess as sp
import os



__AMINOACIDS = frozenset('ACDEFGHIKLMNPQRSTVWYJXOZBU')
__UNKNOWN = {'X', 'O'}

def seq_format(seq):
    return "".join(str(seq).split()).upper()  # Do the minimum formatting

def __calc_ambiguous_values(aa_dict):
    # Rough approximation of weights/values for Ambiguous Amino Acids 
    # (here, we define the weight/value as the mean weight/value between
    # the ambiguous meanings as suggested in
    # https://stackoverflow.com/questions/42159712/biopython-amino-acid-sequence-contains-j-and-cant-calculate-the-molecular-we)
    # 
    # We are using https://www.samformat.info/IUPAC-ambiguity-codes and
    # https://www.dnabaser.com/articles/IUPAC%20ambiguity%20codes.html
    # to define ambiguities.
    
    return {"B" : (aa_dict['D'] + aa_dict['N']) / 2,
            "Z" : (aa_dict['E'] + aa_dict['Q']) / 2,
            "J" : (aa_dict['I'] + aa_dict['L']) / 2}

def molecular_weight(seq, monoisotopic=False):
    # adapted from https://github.com/biopython/biopython/blob/master/Bio/SeqUtils/__init__.py
    seq = seq_format(seq)

    if monoisotopic:
        unambiguous_aa_weights = IUPACData.monoisotopic_protein_weights
        water_weight = 18.010565
    else:
        unambiguous_aa_weights = IUPACData.protein_weights
        water_weight = 18.0153
    
    # rough amino acid weight approximation when monoisotopic is True
    # weight['B'] = (weight['D'] (133.037508) + weight['N'] (132.053492)) / 2
    # weight['Z'] = (weight['E'] (147.053158) + weight['Q'] (146.069142)) / 2
    # weight['J'] = (weight['I'] (131.094629) + weight['L'] (131.094629)) / 2
    #
    # rough amino acid weight when monoisotopic is False
    # weight['B'] = (weight['D'] (133.1027) + weight['N'] (132.1179)) / 2
    # weight['Z'] = (weight['E'] (147.1293) + weight['Q'] (146.1445)) / 2
    # weight['J'] = (weight['I'] (131.1729) + weight['L'] (131.1729)) / 2
    ambiguous_aa_weights = __calc_ambiguous_values(unambiguous_aa_weights)
    
    aa_weights = {**unambiguous_aa_weights, **ambiguous_aa_weights}

    for aa in __UNKNOWN:
        seq = seq.replace(aa, '')
    
    return sum(aa_weights[aa] for aa in seq) - (len(seq) - 1) * water_weight

def instability_index(seq):
    # adapted from https://github.com/biopython/biopython/blob/master/Bio/SeqUtils/ProtParam.py
    # original reference: https://academic.oup.com/peds/article/4/2/155/1491271
    seq = seq_format(seq)

    score = 0.0

    for i in range(len(seq) - 1):
        current_aa, next_aa = seq[i:i+2]

        # considering only valid dipeptides and ignoring the invalid ones
        # as suggested in https://www.biostars.org/p/17833/
        if current_aa in ProtParamData.DIWV and next_aa in ProtParamData.DIWV[current_aa]:
            dipeptide_value = ProtParamData.DIWV[current_aa][next_aa]
            score += dipeptide_value

    return (10.0 / len(seq)) * score

def hydrophobicity(seq):
    # adapted from https://github.com/biopython/biopython/blob/master/Bio/SeqUtils/ProtParam.py
    seq = seq_format(seq)

    # rough kd hydrophobicity approximation for ambiguous values
    # kd['B'] = (kd['D'] (-3.5) + kd['N'] (-3.5)) / 2
    # kd['Z'] = (kd['E'] (-3.5) + kd['Q'] (-3.5)) / 2
    # kd['J'] = (kd['I'] ( 4.5) + kd['L'] ( 3.8)) / 2
    ambiguous_aa_kd = __calc_ambiguous_values(ProtParamData.kd)
    kd = {**ProtParamData.kd, **ambiguous_aa_kd}
    return sum(kd[aa] for aa in seq if aa in kd) / len(seq)

'''Table: Grouping of amino acids based on physiochemical properties. Groups of amino acids with
similar side chains are grouped together to reduce the number of features to test in the machine learning
model (7).
Groups Amino Acids
1 	A, G, V
2 	I, L, F, P
3 	Y, M, T, S
4 	H, N, Q, W
5 	R, K
6	D, E
7 	C
'''

def __group_aminoacids(seq):
    groups = {'A' : '1', 'G' : '1', 'V' : '1',
              # I added 'J' to Group 2, because it may correspond either to 'I' or 'L'.
              # Both are in this group.
              'I' : '2', 'L' : '2', 'F' : '2', 'P' : '2', 'J' : '2',
              'Y' : '3', 'M' : '3', 'T' : '3', 'S' : '3',
              'H' : '4', 'N' : '4', 'Q' : '4', 'W' : '4',
              'R' : '5', 'K' : '5',
              'D' : '6', 'E' : '6',
              'C' : '7',
              # Extra group (Group 0 or "Gap" Group) for ambiguous/uncommon aminoacids (U, X, O, B, and Z)
              'X' : '0', 'O' : '0', 'B' : '0', 'Z' : '0', 'U' : '0'}
    
    return ''.join(groups[s] for s in seq)

def kmers(seq, k, grouped):
    assert k > 0 and k <= len(seq)
    
    if grouped:
        seq = __group_aminoacids(seq)
        keys = [''.join(s) for s in product('12345670', repeat=k)]
    else:
        keys = [''.join(s) for s in product(__AMINOACIDS, repeat=k)]
    
    seq_kmers = OrderedDict.fromkeys(keys, value=0)

    for i in range(len(seq) - k):
        s = seq[i:i+k]
        seq_kmers[s] += 1
    
    return seq_kmers





def analysis(sample_sequence):


        analyse = ProteinAnalysis(str(sample_sequence[:-1]).replace('U', 'S'))
        aa_dict = analyse.count_amino_acids()
        
        num_amino_acids = '%d'%(sum(list(aa_dict.values())))
        molecular_weight = '%.3f'%(analyse.molecular_weight())
        pI = '%.3f'%(analyse.isoelectric_point())
        
        neg_charged_residues = '%d'%(aa_dict['D'] + aa_dict['E'])
        pos_charged_residues = '%d'%(aa_dict['K'] + aa_dict['R'])
        
        extinction_coefficients_1 = '%d'%(aa_dict['Y']*1490 + aa_dict['W']*5500)
        extinction_coefficients_2 = '%d'%(aa_dict['Y']*1490 + aa_dict['W']*5500 + aa_dict['C']*125)
        
        instability_index = '%.3f'%(analyse.instability_index())
        gravy = '%.3f'%(analyse.gravy())
        
        secondary_structure_fraction = tuple(['%.3f'%(frac) for frac in analyse.secondary_structure_fraction()])
        
        analyses = [num_amino_acids, molecular_weight, pI, 
                         neg_charged_residues, pos_charged_residues,
                         extinction_coefficients_1, extinction_coefficients_2,
                         instability_index, gravy, 
                         *secondary_structure_fraction]



        return [float(a) for a in analyses]






class MyOwnDataset():

    def __init__(self, root: str, input_files: list, max_norm: list = [None], max_length: list = [None], shuffle: bool = True, start_label: int = 0, seq_complete: str = "None", DNA: bool = False):

        self.shuffle = shuffle
        self.max_norm = max_norm
        self.file_list = input_files
        self.path_way = root
        self.num_features = 21
        self.prot_start_label = start_label
        self.mean = []
        self.std = []
        self.max_length = 0
        self.max_length_input = max_length
        self.filenames = [self.path_way + file for file in input_files]
        self.class_length = len(input_files)
        self.sequence_completeness = seq_complete
        self.mode_DNA = DNA




    def prodigal(self, prodigal_cmd, fasta_file, completeness):
        meta = ' -p meta ' if completeness == 'partial' else ''
        fasta_file_preffix = fasta_file.rsplit('.', 1)[0]
        output_fasta_file = fasta_file_preffix + '_proteins.fa'
        log_file = fasta_file_preffix + '_prodigal.log'
        prodigal_cmd += ' -i {input_fasta}  -c -m -g 11 -a {output_fasta} -q' + meta
        prodigal_cmd = prodigal_cmd.format(prodigal=prodigal_cmd, input_fasta=fasta_file, output_fasta=output_fasta_file)

        with open(log_file, 'w') as lf:
            sp.call(prodigal_cmd.split(), stdout=lf)

        return output_fasta_file



    
    
    def create_fasta(self, input_file, sequence_completeness):
    
    
        fasta_file = self.prodigal("prodigal", input_file, sequence_completeness)
    
        return fasta_file


    def process(self):


            data_list = []
            protein_label = self.prot_start_label
            z_list = []
            max_length_list = []
            data_strings = []

            for file_name in self.filenames:
                

                if self.mode_DNA == True:
                
                    file_name = self.create_fasta(file_name, self.sequence_completeness)


            
                data_, max_length = self.load_protein_data(file_name)
                data_strings.append(data_)
                max_length_list.append(max_length)

            max_length_list = [self.max_length_input] if np.array(self.max_length_input).all() != None else max_length_list

            for prot_data in data_strings:
                for c in range(len(prot_data[0])):
                    graph_data, z_data = self.create_graph(protein_label,[prot_data[0][c],prot_data[1][c]], max_ =  max_length_list)
                    z_list.append(z_data)
                    data_list.append(graph_data)
                protein_label = protein_label + 1

            max_data = np.max(z_list, axis=0) if np.array(self.max_norm).all() == None else self.max_norm
            data_list = self.adjust_zdata(data_list, max_data)

            if self.shuffle:
                random.shuffle(data_list)

            self.max_length =np.max(max_length_list)


            return data_list,max_data




    def load_protein_data(self, file: str):

        protein_list_full = []
        protein_id_full = []
        protein_list = []
        protein_id = []

        try:

            for record in SeqIO.parse(file, "fasta"):

                if any(SeqIO.parse(file, "fasta"))  == False:
                    print("Files need to have fasta format")
                    sys.exit(1)

                protein_list.append(str(record.seq))
                protein_id.append(str(record.id))

        except FileNotFoundError:
            print("File " + file + " not accessible")
            sys.exit(0)

        protein_list_full.extend(protein_list)
        protein_id_full.extend(protein_id)
        max_len = np.max([len(x) for x in protein_list_full])

        return [protein_list_full, protein_id_full], max_len




    def adjust_zdata(self, data_list:list, max_data:list):

        data_list = [[data[0], data[1],data[2]/max_data, data[3],data[4]] for data in data_list]
        data_list = [Data(x=data[0], y=data[1], z= data[2].float(), edge_index=data[3].t().contiguous(), id = data[4]) for data in data_list]


        return data_list




    def read_yaml(self):

        try:
            #with open(r'/home/fr/fr_fr/fr_sh765/dec2.models_save_method_bc/aminoacids.yaml') as file:
            with open(r'./aminoacids.yaml') as file:
              #with open(r'/home/fr/fr_fr/fr_sh765/method/aminoacids.yaml') as file:
                documents = yaml.full_load(file)
        except FileNotFoundError:
            print("File aminoacids.yaml not accessible")
            sys.exit(0)

        self.num_features = len(documents)

        return documents



    def create_one_hot_vector(self, string_graph, dic_amino):
        
        onehot_encoded = []
        
        try:
            size = len(dic_amino.keys())
            for char in string_graph:
                    val = dic_amino.get(char)
                    listofzeros = [0] * size
                    listofzeros[val-1] = 1
                    onehot_encoded.append(listofzeros)

        except TypeError:
            print("Aminoacid abbreviation " + char + " not found in aminoacid.yaml")
            sys.exit(2)
            

        return onehot_encoded
    
    
    def zeropad_or_cut(self, onehot_encoded, max_length, vector_size):
    
        onehot_encoded = onehot_encoded[:max_length]

        to_pad = max_length - len(onehot_encoded)
        for i in range(to_pad):
            listofzeros = [0] * vector_size
            onehot_encoded.append(listofzeros)
            
    
        return onehot_encoded


    def create_graph(self, prot_type,protein, max_ = None):
        dic_amino = self.read_yaml()
        size = len(dic_amino.keys())
        max_length = np.max(max_)

        edge_list = []

        prot_id, string_graph = protein[1], protein[0]
        l_length = len(string_graph)
        
        if string_graph.endswith("*"): string_graph = string_graph[:-1]
        z_data = analysis(string_graph)
        

        for it in range(0, len(string_graph) -1):
            edge_list.append([it, it+1])
            edge_list.append([it+1, it])

        onehot_encoded = self.create_one_hot_vector(string_graph, dic_amino)
        onehot_encoded = self.zeropad_or_cut(onehot_encoded, max_length, size)
        

        
        edge_index = torch.tensor(edge_list, dtype=torch.long)
        x = torch.tensor(onehot_encoded, dtype=torch.float)
        y = torch.tensor([prot_type], dtype=torch.int)
        z = torch.tensor(z_data,dtype = torch.float)

        data = [x,y,z,edge_index,str(prot_id)]

        return data, z_data

