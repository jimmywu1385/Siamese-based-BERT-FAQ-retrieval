import random
import time
import datetime
import os
traindata=[]
faqquestion=[]
faqanswer=[]
##########################################################
with open("stackExchange-FAQ.xml","r") as f:
    f.readline() #read rootflag
    for i in range(125):
        f.readline() #read qapairflag
        flag=f.readline()
        data=[]
        while flag=='<rephr>\n':
            d=f.readline()
            if d!='*\n':
                data.append(d)
            f.readline() #read <\rephr>
            flag=f.readline()
        traindata.append(data)
        query=f.readline()
        faqquestion.append(query) #read question flag
        f.readline() #read question flag
        flag=f.readline()
        aset=[]
        while flag=='<answer>\n':
            answer=f.readline()
            aset.append(answer)
            f.readline()
            flag=f.readline()
        faqanswer.append(aset)
    f.readline() #read rootflag
##############################################################
with open("traindata.txt","w") as f:
    for i in range(125):
        for e in traindata[i]:
            line=str(i)+' '+e
            f.writelines(line)
###############################################################
with open("faq.txt","w") as f:
    for i in range(125):
        for e in faqanswer[i]:
            line=str(i)+' '+faqquestion[i]+e
            f.writelines(line)
###############################
seed_val = 42
random.seed(seed_val)
##########read file###########
faqset=[]
with open("faq.txt","r") as f:
    for i in range(719):
        faqsubset=[]
        lines=f.readline()
        if len(lines)==0 or lines=='\n':
            break       
        line=[e for e in lines.split()]
        faqquestion=" ".join(line[1:])
        faqanswer=f.readline()
        faqsubset.append([i,int(line[0]),faqquestion,faqanswer])
        faqset.append(faqsubset)
        
querydata=[]
with open("traindata.txt","r") as f:
    while 1:
        lines=f.readline()
        if len(lines)==0:
            break
        line=[e for e in lines.split()]
        query=" ".join(line[1:])
        querydata.append([int(line[0]),query])
##########split data################
random.shuffle(querydata)
train_data=querydata[:750]
valid_data=querydata[750:1000]
test_data=querydata[1000:]
#########store data###############
with open("train_set.txt","w") as f:
    for i in range(750):
        line=str(train_data[i][0])+' '+train_data[i][1]
        f.writelines(line)
        f.writelines("\n")
with open("valid_set.txt","w") as f:
    for i in range(250):
        line=str(valid_data[i][0])+' '+valid_data[i][1]
        f.writelines(line)        
        f.writelines("\n")
with open("test_set.txt","w") as f:
    for i in range(250):
        line=str(test_data[i][0])+' '+test_data[i][1]
        f.writelines(line)
        f.writelines("\n")
