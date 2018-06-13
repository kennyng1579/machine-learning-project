import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('../data/training.csv')

horse_id = input("Enter horse_id: ")


pos = data.loc[data['horse_id'] == horse_id].finishing_position[-6:].values
date = data.loc[data['horse_id'] == horse_id].race_id[-6:].values

plt.figure(figsize=(8, 8))
plt.plot(date, pos, color='blue', linewidth=3)
plt.ylim([0,15])

plt.xlabel('recent games', size=8)
plt.ylabel('finishing position', size=8)
plt.title('Recent Racing Result of '+horse_id, size=15)
plt.savefig('line_chart.png')

plt.show()