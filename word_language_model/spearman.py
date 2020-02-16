import torch
import pandas as pd 
import data
from scipy.stats import spearmanr
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
args = parser.parse_args()

word_pairs = pd.read_csv('wordsim353/combined.csv', sep = ',')
word_1 = word_pairs['Word 1'].values
word_2 = word_pairs['Word 2'].values

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f)
model.eval()

corpus = data.Corpus(args.data)

total_coef = 0. 
for idx in range(len(word_pairs)):
    if word_1[idx] in corpus.dictionary.word2idx and word_2[idx] in corpus.dictionary.word2idx:
        word_1_idx = corpus.dictionary.word2idx[word_1[idx]]
        word_2_idx = corpus.dictionary.word2idx[word_2[idx]]

        word_1_vec = model.emb.weight[word_1_idx].detach().numpy()
        word_2_vec = model.emb.weight[word_2_idx].detach().numpy()
        
        coef, p = spearmanr(word_1_vec, word_2_vec)
        total_coef += coef

print('Spearman Corr: {:5.4f}'.format(total_coef/len(word_pairs)))

