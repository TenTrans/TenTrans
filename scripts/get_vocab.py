import sys
import  argparse
from collections import Counter


parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("--special_tokens", nargs="+", default=[]) 

# mask: [mask]
#lang_embed: [en_embed] [fr_embed]
params = parser.parse_args()
total_speical = 200

file = open(params.input, "r", encoding='utf8')
c = Counter()

for spec in params.special_tokens:
    c[spec] = 10**7

for i in range(len(params.special_tokens), total_speical):
    c[f'[spec_{i}]'] = 10**7

for line in file:
    for word in line.strip().split(' '):
        if word:
            c[word] += 1

for k, v in sorted(c.items(), key=lambda x: x[1], reverse=True):
    print(f"{k} {v}")