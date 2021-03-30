import torch
from transformers import *
import torch.nn as nn
from sklearn import decomposition
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
output_dir = './model_idf2'
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

##########read file###########
faqset=[]
with open("faq_threadQ.txt","r") as f:
    while 1:
        lines=f.readline()
        if len(lines)==0:
            break
        line=[e for e in lines.split()]
        faqquestion=" ".join(line[1:])
        faqset.append([int(line[0]),faqquestion])
##################encode data#################
faq_ids=[]
faq_masks=[]
for i in range(125):
    encoded_dict1 = tokenizer.encode_plus(
                        faqset[i][1],                    
                        add_special_tokens = True, 
                        max_length = 100,          
                        pad_to_max_length = True,
                        return_attention_mask = True,   
                        return_tensors = 'pt',     
                   )   
    faq_ids.append(encoded_dict1['input_ids'])
    faq_masks.append(encoded_dict1['attention_mask'])
faq_ids = torch.cat(faq_ids, dim=0)
faq_masks = torch.cat(faq_masks, dim=0)

output = model(faq_ids, faq_masks,'idf')

###############show data##########################
output=output.tolist()
pca = decomposition.PCA(n_components=3)
pca.fit(output)
output = pca.transform(output)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in output:
    print(i)
    ax.scatter(i[0],i[1],i[2],s=10, c='b', marker='^')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
