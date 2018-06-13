import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('../data/training.csv')

horse_ids = data.horse_id.unique()
jockeys = data.jockey.unique()

horse_wins = []
horse_win_rates = []
jockey_wins = []
jockey_win_rates = []

for x in data.horse_id.unique():
    num_win = len(data.loc[data['horse_id']==x].loc[data['finishing_position']==1].values)
    num_play = len(data.loc[data['horse_id']==x].values)
    win_rate = num_win/num_play
    horse_wins.append(num_win)
    horse_win_rates.append(win_rate)

for x in data.jockey.unique():
    num_win = len(data.loc[data['jockey']==x].loc[data['finishing_position']==1].values)
    num_play = len(data.loc[data['jockey']==x].values)
    win_rate = num_win/num_play
    jockey_wins.append(num_win)
    jockey_win_rates.append(win_rate)

plt.figure(figsize=(11, 11))
plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)

plt.subplot(2,1,1)
plt.title('Win rate verus number of wins for horses', size=15)
plt.scatter(horse_win_rates, horse_wins, c='red', alpha=0.5)
plt.scatter([x for x,y in zip(horse_win_rates,horse_wins) if x>=0.5 and y>=5], [y for x,y in zip(horse_win_rates,horse_wins) if x>=0.5 and y>=5], c='cyan')
plt.xlabel('Win rate', size=11)
plt.ylabel('Number of wins', size=11)
for i, txt in enumerate(horse_ids):
    if(horse_win_rates[i]>=0.5 and horse_wins[i]>=5):
        plt.annotate(txt, (horse_win_rates[i], horse_wins[i]))

plt.subplot(2,1,2)
plt.title('Win rate verus number of wins for jockeys', size=15)
plt.scatter(jockey_win_rates, jockey_wins, c='blue', alpha=0.5)
plt.scatter([x for x,y in zip(jockey_win_rates,jockey_wins) if x>=0.1 and y>=100], [y for x,y in zip(jockey_win_rates,jockey_wins) if x>=0.1 and y>=100], c='green')
plt.xlabel('Win rate', size=11)
plt.ylabel('Number of wins', size=11)
for i, txt in enumerate(jockeys):
    if(jockey_win_rates[i]>=0.1 and jockey_wins[i]>=100):
        plt.annotate(txt, (jockey_win_rates[i], jockey_wins[i]))

plt.savefig('scatter_chart.png')
plt.show()