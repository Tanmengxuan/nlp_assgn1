import torch
import pandas as pd 
import data
from scipy.stats import spearmanr
from scipy.spatial import distance
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
word_sim = word_pairs['Human (mean)'].values

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f)
model.eval()

corpus = data.Corpus(args.data)

human_sim = []
emb_sim = []
for idx in range(len(word_pairs)):
    if word_1[idx] in corpus.dictionary.word2idx and word_2[idx] in corpus.dictionary.word2idx:
        word_1_idx = corpus.dictionary.word2idx[word_1[idx]]
        word_2_idx = corpus.dictionary.word2idx[word_2[idx]]

        word_1_vec = model.emb.weight[word_1_idx].detach().cpu().numpy()
        word_2_vec = model.emb.weight[word_2_idx].detach().cpu().numpy()
        
        cos_sim = 1 - distance.cosine(word_1_vec, word_2_vec)
        emb_sim.append(cos_sim)
        human_sim.append(word_sim[idx])        

coef, p = spearmanr(emb_sim, human_sim)
print('Spearman Corr: {:5.4f}'.format(coef))

