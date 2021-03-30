from sentence_transformers import SentenceTransformer, util
import json
import torch
from transformers import *
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import math
output_dir = './model_cls'
idf={}
with open('idf.txt','r') as f:
    while 1:
        line=f.readline()
        if len(line)==0 or line=='\n':
            break 
        line=[float(e) for e in line.split()]
        idf[line[0]]=line[1]

class Sbert(nn.Module):
    def __init__(self,idf):
        super(Sbert, self).__init__()
        self.bert= BertModel.from_pretrained(output_dir)
        self.idf=idf
    def forward(self, in1,in1m,pooling='idf'):
        loss1, a = self.bert(in1, 
                             token_type_ids=None, 
                             attention_mask=in1m)
#################pooling###########################
        if pooling=='idf':
            for i in range(len(in1)):
                for j in range(100):
                    if in1m[i][j]==1:
                        idf_weight=0.0
                        if int( in1[i][j]) in self.idf:
                            idf_weight=math.log(719/(1+self.idf[int(in1[i][j])]),2)
                        else:
                            idf_weight=math.log(719/1,2)
                        loss1[i][j]*=idf_weight

            input_mask_expanded1 = in1m.unsqueeze(-1).expand(loss1.size()).float()
            sum_embeddings1 = torch.sum(loss1 * input_mask_expanded1, 1)
            sum_mask1 = torch.clamp(input_mask_expanded1.sum(1), min=1e-9)
            output_vector1 = sum_embeddings1 / sum_mask1

        if pooling=='avg':
            input_mask_expanded1 = in1m.unsqueeze(-1).expand(loss1.size()).float()
            sum_embeddings1 = torch.sum(loss1 * input_mask_expanded1, 1)
            sum_mask1 = torch.clamp(input_mask_expanded1.sum(1), min=1e-9)
            output_vector1 = sum_embeddings1 / sum_mask1
        
#[cls]token#
        if pooling=='cls':
            output_vector1=loss1[:, 0, :].float() 
#max#
        if pooling=='max':
            input_mask_expanded1 = in1m.unsqueeze(-1).expand(loss1.size()).float()
            loss1[input_mask_expanded1 == 0] = -1e9 
            output_vector1 = torch.max(loss1, 1)[0]

        return output_vector1

tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
model=Sbert(idf)

FAQ=[]

##load faq and testset
with open("faq.txt","r") as f:
    for i in range(719):
        line=f.readline()
        line=[e for e in line.split()]
        question=" ".join(line[1:])
        answer=f.readline()
        FAQ.append([i,int(line[0]),question,answer])
testdata=[]
with open("test_set.txt","r") as f:
    for i in range(250):
        line=f.readline()
        line=[e for e in line.split()] 
        query=" ".join(line[1:])
        testdata.append([int(line[0]),query])  

##make score board        
scoreboard=[]
for i in range(250):
    encoded_dict1=tokenizer.encode_plus(
                testdata[i][1],                    
                add_special_tokens = True, 
                max_length = 100,          
                pad_to_max_length = True,
                return_attention_mask = True,   
                return_tensors = 'pt', 
                truncation=True    
           )
    invector=model(encoded_dict1['input_ids'],encoded_dict1['attention_mask'],'cls')
    faq_score=[]
    for j in range(719):
        encoded_dict2=tokenizer.encode_plus(
                    FAQ[j][2],                    
                    add_special_tokens = True, 
                    max_length = 100,          
                    pad_to_max_length = True,
                    return_attention_mask = True,   
                    return_tensors = 'pt',  
                    truncation=True   
               )
        faqvector=model(encoded_dict2['input_ids'],encoded_dict2['attention_mask'],'cls')  
        score=float(torch.cosine_similarity(faqvector,invector))
        faq_score.append([FAQ[j][1],score])
        print(i,j)
    scoreboard.append(faq_score)

##sort each score
for i in range(250):
    scoreboard[i].sort(key=lambda s: s[1],reverse=True)
    
path='log_cls/query'
for i in range(250):
    with open(path+str(i)+'.txt','w') as f:
        json.dump(scoreboard[i],f)

##mrr map p@5
mrr=0.0
p5=0.0
map=0.0

for i in range(250):
    count=0.0
    ap=0.0
    target_faq=testdata[i][0]
    first=True
    for j in range(719):
        if scoreboard[i][j][0]==target_faq:
            count+=1
            ap+=count/(j+1)
            if first:
                mrr+=1/(j+1)
                first=False
        if j==4:
            p5+=count/5
    map+=(ap/count)

map/=250
p5/=250
mrr/=250
print(map,p5,mrr)
with open(path+'result.txt','w') as f:
    f.writelines(str(map)+'\n')
    f.writelines(str(p5)+'\n')
    f.writelines(str(mrr)+'\n')