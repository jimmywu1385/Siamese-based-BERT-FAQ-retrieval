import torch
from transformers import BertTokenizer 
import math
tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
FAQ=[]
with open("faq.txt","r") as f:
    for i in range(719):
        line=f.readline()
        line=[e for e in line.split()]
        question=" ".join(line[1:])
        answer=f.readline()
        tokens1 = tokenizer.tokenize(question)
        ids1 = tokenizer.convert_tokens_to_ids(tokens1)
        tokens2 = tokenizer.tokenize(answer)
        ids2 = tokenizer.convert_tokens_to_ids(tokens2)
        FAQ.append(ids1+ids2)

term={}
for i in range(719):
    for e in FAQ[i]:
        if e not in term:
            term[e]=0.0
    for j in range(len(FAQ[i])):
            if FAQ[i][j] not in FAQ[i][:j]:
                term[FAQ[i][j]]+=1.0   

with open('idf.txt','w') as f:
    for i in term:
        f.writelines(str(i)+' '+str(term[i])+'\n')
