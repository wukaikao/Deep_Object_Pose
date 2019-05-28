import pandas as pd 
import os
	
now_work_path = os.getcwd()
print(now_work_path)
df = pd.read_csv(str(now_work_path)+'/train_1_v2_2_stage3_/loss_train.csv') 
df2 = pd.read_csv(str(now_work_path)+'/train_1_v2_2_stage3_/loss_test.csv') 

epoch = 7
count  =[]
average=[]
average2=[]
for i in range(epoch):
    count.append(0)
    average.append(0)
    average2.append(0)

for i in range(len(df['epoch'])) :
    for epoch_num in range(epoch):
        if df['epoch'][i]==epoch_num+1:
            print("epoch : "+str(epoch_num))
	    count[epoch_num] = count[epoch_num]+1
            average[epoch_num]= df['loss'][i] + average[epoch_num]

for i in range(len(df2['epoch'])) :
    for epoch_num in range(epoch):
        if df2['epoch'][i]==epoch_num+1:
            print("epoch : "+str(epoch_num))
	    count[epoch_num] = count[epoch_num]+1
            average2[epoch_num]= df2['loss'][i] + average2[epoch_num]


data_dict = {"train_epoch":[],"train_loss":[],"test_epoch":[],"test_loss":[]} 

for i in range(epoch):
    data_dict["train_epoch"].append(i+1)
    data_dict["train_loss"].append(average[i]/count[i])
    data_dict["test_epoch"].append(i+1)
    data_dict["test_loss"].append(average2[i]/count[i])


data_df = pd.DataFrame(data_dict)
data_df.to_csv('train_1_v2_2_stage3_.csv')
