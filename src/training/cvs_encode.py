import pandas as pd 
import os
import sys


__, train_folder,epoch = sys.argv
print(train_folder,epoch)
now_work_path = os.getcwd()
df = pd.read_csv(str(now_work_path)+"/"+str(train_folder)+'loss_train.csv') 
df2 = pd.read_csv(str(now_work_path)+"/"+str(train_folder)+'loss_test.csv') 

# epoch = 7
epoch = int(epoch)
count  =[]
average=[]
average2=[]
for i in range(epoch):
    count.append(0)
    average.append(0)
    average2.append(0)

print("loss_train compressing......")
for i in range(len(df['epoch'])) :
    for epoch_num in range(epoch):
        if df['epoch'][i]==epoch_num+1:
            # print("epoch : "+str(epoch_num))
	    count[epoch_num] = count[epoch_num]+1
            average[epoch_num]= df['loss'][i] + average[epoch_num]

print("loss_test comressing......")
for i in range(len(df2['epoch'])) :
    for epoch_num in range(epoch):
        if df2['epoch'][i]==epoch_num+1:
            # print("epoch : "+str(epoch_num))
	    count[epoch_num] = count[epoch_num]+1
            average2[epoch_num]= df2['loss'][i] + average2[epoch_num]


data_dict = {"train_epoch":[],"train_loss":[],"test_epoch":[],"test_loss":[]} 

for i in range(epoch):
    data_dict["train_epoch"].append(i+1)
    data_dict["train_loss"].append(average[i]/count[i])
    data_dict["test_epoch"].append(i+1)
    data_dict["test_loss"].append(average2[i]/count[i])


data_df = pd.DataFrame(data_dict)
data_df.to_csv(str(now_work_path)+"/"+str(train_folder)+'light_loss.ods')
print("FINISH!\n[light_loss.csv] is saving in "+str(now_work_path)+"/"+str(train_folder))
