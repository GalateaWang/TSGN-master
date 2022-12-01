# TSGN-master        ![github](https://img.shields.io/badge/github-GalateaWang-brightgreen.svg) ![Github stars](https://img.shields.io/github/stars/GalateaWang/TSGN-master.svg) ![Mozilla Add-on](https://img.shields.io/amo/dw/:addonId) ![thanks author](https://img.shields.io/badge/github-PengtaoChen-green.svg)

For the paper "TSGN: Trandaction Subgraph Networks Assisting Phishing Detection in Ethereum" [Arxiv Link](https://arxiv.org/pdf/2208.12938.pdf)

Due to the decentralized and public nature of the Blockchain ecosystem, the malicious activities on the Ethereum platform impose immeasurable losses for the users. Existing phishing scam detection methods mostly rely only on the analysis of original transaction networks, which is difficult to dig deeply into the transaction patterns hidden in the network structure of transaction interaction. In this paper, we propose a Transaction SubGraph Network (TSGN) based phishing accounts identification framework for Ethereum. We first extract transaction subgraphs for target accounts and then expand these subgraphs into corresponding TSGNs based on the different mapping mechanisms. In order to make our model incorporate more important information about real transactions, we encode the transaction attributes into the modeling process of TSGNs, yielding two variants of TSGN, i.e., Directed-TSGN and Temporal-TSGN, which can be applied to the different attributed networks. Especially, by introducing TSGN into multi-edge transaction networks, the proposed Multiple-TSGN model is able to preserve the temporal transaction flow information and capture the significant topological pattern of phishing scams, while reducing the time complexity of modeling large-scale networks. Extensive experimental results show that TSGN models can provide more potential information to improve the performance of phishing detection by incorporating graph representation learning.

## Rquirements

- python--3.9.0
- pytorch--1.12.0
- gensim--3.8.0
- pandas
- numpy
- tqdm
- pickle
- networkx
- scikit-learn

## Usage

Execute the following commands in the same directory where the code resides:

```python
python SGN.py  # obtain the TSGNs
python baseline.py  # excute the corresponding basline algorithm for obtaining the transaction ego network embeddings 
python baseline-classification.py  # phishing account identification
```




## Cite

```
@article{wang2022tsgn,
  title={TSGN: Transaction Subgraph Networks Assisting Phishing Detection in Ethereum},
  author={Wang, Jinhuan and Chen, Pengtao and Xu, Xinyao and Wu, Jiajing and Shen, Meng and Xuan, Qi and Yang, Xiaoniu},
  journal={arXiv preprint arXiv:2208.12938},
  year={2022}
}
```
