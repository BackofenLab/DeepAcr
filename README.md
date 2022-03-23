# Anti-CRISPR prediction using deep learning reveals an inhibitor of Cas13b nucleases

As part of the ongoing bacterial-phage arms race, CRISPR-Cas systems in bacteria clear invading phages while anti-CRISPR proteins (Acrs) in phages inhibit CRISPR defenses. Known Acrs have proven extremely diverse, complicating their identification. Here, we report a deep learning algorithm for Acr identification that revealed an Acr against  CRISPR-Cas systems. The algorithm predicted numerous putative Acrs spanning almost all CRISPR-Cas types and sub-types

### First step: download the last version of the tool and extract it


```
wget https://github.com/BackofenLab/Acr/archive/1.0.0.tar.gz
tar -xzf 1.0.0.tar.gz

### Second step: (conda)

First we install Miniconda for python 3.
Miniconda can be downloaded from here: [miniconda](https://docs.conda.io/en/latest/miniconda.html).

Install Miniconda.

``
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
