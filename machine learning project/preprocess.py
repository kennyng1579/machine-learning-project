import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#def readfile (filepath):
df = pd.read_csv('./data/race-result-horse.csv')
df1 = pd.read_csv('./data/race-result-race.csv')
#df.info()
fin_pos = df.finishing_position

#Locate useless data and adjust data
drop_rows = []
counter = 0
for data in fin_pos:
    try:
        data = float(data)
    except ValueError:
        if(" DH" in data):
            new_data = data[:-3]
            new_data = int(new_data)
            df['finishing_position'] = df['finishing_position'].replace(data, new_data)
            #print (new_data)
        else:
            drop_rows.append(counter)
        #print(data)
    counter+=1
print("Number of rows drop: ", len(drop_rows))

#Drop useless data
df = df.drop(df.index[drop_rows])
df = df.dropna(subset=['finishing_position'])


#Handling jockey and trainer average rank
horse_ids = df.horse_id.unique().tolist()
jockeys = df.jockey.unique().tolist()
trainers = df.trainer.unique().tolist()

print("# of horse: ", len(horse_ids))
print("# of jockeys: ", len(jockeys))
print("# of trainers: ", len(trainers))

print("Processing horse, jockeys and trainers average rank...")
jockey_rank = []
trainer_rank = []

for data in jockeys:
    x = df.loc[df['jockey'] == data].loc[df['race_id']<"2016-328"].finishing_position.values
    ranking = list(map(int, x))
    if(len(ranking) == 0):
        jockey_rank.append(7)
    else:
        jockey_rank.append(int(sum(ranking)/len(ranking)))

for data in trainers:
    x = df.loc[df['trainer'] == data].loc[df['race_id']<"2016-328"].finishing_position.values
    ranking = list(map(int, x))
    if(len(ranking) == 0):
        trainer_rank.append(7)
    else:
        trainer_rank.append(int(sum(ranking)/len(ranking)))
print("Done")
  

#Handling recent average rank of each horse
print("Add index, distance and calculatin recent average rank for each horse...")
horse_index = []
jockey_index = []
trainer_index = []
jockey_ave_rank = []
trainer_ave_rank = []
race_distance = []
rank6 = []
recent_ave_rank = []
HorseWin = []
HorseRankTop3 = []
HorseRankTop50Percent = []
counter = 0

for data in df.itertuples():
    #labels for classification
    if(int(data.finishing_position)==1):
        HorseWin.append(1)
    else:
        HorseWin.append(0)
        
    if(int(data.finishing_position)<=3):
        HorseRankTop3.append(1)
    else:
        HorseRankTop3.append(0)
    num = df.loc[df['race_id']==data.race_id].shape[0]
    #print(num)
    if(int(data.finishing_position)<= num/2):
        HorseRankTop50Percent.append(1)
    else:
        HorseRankTop50Percent.append(0)
    
    #Index of horses, jockeys and trainers
    in_horse = horse_ids.index(data.horse_id)
    in_jockey = jockeys.index(data.jockey)
    in_trainer = trainers.index(data.trainer)
    horse_index.append(in_horse)
    jockey_index.append(in_jockey)
    trainer_index.append(in_trainer)
    jockey_ave_rank.append(jockey_rank[in_jockey])
    trainer_ave_rank.append(trainer_rank[in_trainer])
    
    #Distance
    distance = df1.loc[df1['race_id'] == data.race_id].race_distance.values
    race_distance.append(int(distance))
    
    #Recent horse rank
    pos = df.loc[df['horse_id'] == data.horse_id].loc[df['race_id'] <= data.race_id].finishing_position[-6:]

    recent_6 = list(map(int, pos.values))
    rank6.append(recent_6)
    if(len(recent_6)==1):
        recent_ave_rank.append(7)
    else:
        recent_ave_rank.append(int(sum(recent_6)/len(recent_6)))

    counter+=1
    if(counter%1000==0):
        print("%d steps..."%counter)

df['HorseWin'] = HorseWin
df['HorseRankTop3'] = HorseRankTop3
df['HorseRankTop50Percent'] = HorseRankTop50Percent
df['horse_index'] = horse_index
df['jockey_index'] = jockey_index
df['trainer_index'] = trainer_index
df['jockey_ave_rank'] = jockey_ave_rank
df['trainer_ave_rank'] = trainer_ave_rank
df['race_distance'] = race_distance
df['recent_ave_rank'] = recent_ave_rank
print("Done")

#Split to training and testing data
print ("Split dataset and write to different files...")
training_set = df.loc[df['race_id']<"2016-328"]
testing_set = df.loc[df['race_id']>="2016-328"]
training_set.to_csv('./data/training.csv')
testing_set.to_csv('./data/testing.csv')
print ("Done")

print("Preprocess done!")