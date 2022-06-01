# Anti-CRISPR prediction using deep learning reveals an inhibitor of Cas13b nucleases

As part of the ongoing bacterial-phage arms race, CRISPR-Cas systems in bacteria clear invading phages while anti-CRISPR proteins (Acrs) in phages inhibit CRISPR defenses. Known Acrs have proven extremely diverse, complicating their identification. Here, we report a deep learning algorithm for Acr identification that revealed an Acr against  CRISPR-Cas systems. The algorithm predicted numerous putative Acrs spanning almost all CRISPR-Cas types and sub-types

### Installation and requirements

DeepAcr_masterscript.py has been tested with Python 3.7 To run it, we recommend installing the same library versions we used. Since we exported our classifiers following the [model persistence guideline from scikit-learn](https://scikit-learn.org/stable/modules/model_persistence.html), it is not guaranteed that they will work properly if loaded using other Python and/or library versions. For such, we recommend the use a conda virtual environment. They make it easy to install the correct Python and library dependencies without affecting the whole operating system (see below).



### First step: download the last version of the tool and extract it


```
git clone git@github.com:BackofenLab/Acr.git

OR 

https://github.com/BackofenLab/Acr/archive/refs/heads/main.zip
unzip main.zip

```

### Second step: download the deep Learning (ML) models

Due to GitHub's file size constraints, we made our DL models available in Google Drive. You can download them  [here](https://drive.google.com/f). Save both tar.gz files inside modules directory.


### Third step: (conda)

First we install Miniconda for python 3.
Miniconda can be downloaded from here: [miniconda](https://docs.conda.io/en/latest/miniconda.html).

Install Miniconda.

``
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
``

Create and activate environment for DeepAcr.

```
conda env create -f environment.yml -n DeepAcr-env
conda activate DeepAcr-env
```
### Quick run with the default parameters

```
python3.7 DeepAcr_masterscript -f <file> -m <path_to_models> -d <path_to_data>
 

```
## Citation
If you use our tool, please cite our article
- [Anti-CRISPR prediction using deep learning reveals an inhibitor of Cas13b nucleases ](https://www.cell.com/molecular-cell/fulltext/S1097-2765(22)00437-3)
  Katharina G. Wandera, Omer S. Alkhnbashi, Harris V.I. Bassett, Alexander Mitrofanov, Sven Hauns, Anzhela Migur, Rolf Backofen, Chase L. Beisel, 
  Molecular Cell, May 31, 2022, DOI(https://doi.org/10.1016/j.molcel.2022.05.003 ).
