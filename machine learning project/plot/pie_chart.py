import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('../data/training.csv')

sizes = [0]*14
for x in data.race_id.values:
    rank = data.loc[data['race_id']==x].loc[data['finishing_position']==1].draw.values
    if(len(rank)>1):
        for i in rank:
             sizes[int(i)-1] += 1
    else:
        sizes[int(rank)-1] += 1

colors = ['aqua', 'azure', 'beige', 'blue', 'brown', 'chartreuse', 'coral', 'cyan', 'darkgreen', 'fuchsia', 'gold', 'green', 'indigo', 'khaki']
labels = ['Draw 1', 'Draw 2', 'Draw 3', 'Draw 4', 'Draw 5', 'Draw 6', 'Draw 7' ,'Draw 8' ,'Draw 9', 'Draw 10', 'Draw 11', 'Draw 12', 'Draw 13', 'Draw 14']

plt.figure(figsize=(8, 8))
plt.title('Draw Bias Effect (Winning percentage of each draw)', size=15)
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.savefig('pie_chart.png')
plt.show()