from Bio.Data import IUPACData
from Bio.SeqUtils import ProtParamData 
                                       

from collections import OrderedDict
from itertools import product
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import pandas as pd
import argparse
from Bio import SeqIO


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
    
    
    ambiguous_aa_weights = __calc_ambiguous_values(unambiguous_aa_weights)
    
    aa_weights = {**unambiguous_aa_weights, **ambiguous_aa_weights}

    for aa in __UNKNOWN:
        seq = seq.replace(aa, '')
    
    return sum(aa_weights[aa] for aa in seq) - (len(seq) - 1) * water_weight

def instability_index(seq):
    
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









