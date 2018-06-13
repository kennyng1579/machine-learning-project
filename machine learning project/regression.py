import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from operator import itemgetter
import time

train = pd.read_csv('./data/training.csv')
test = pd.read_csv('./data/testing.csv')

finish_time_train = []
finish_time_test = []

for data in train.itertuples():
    cur_time = data.finish_time
    time_ary = cur_time.split('.')
    abs_time = int(time_ary[0])*60+int(time_ary[1])+float(time_ary[2])/100
    finish_time_train.append(abs_time)
    
for data in test.itertuples():
    cur_time = data.finish_time
    time_ary = cur_time.split('.')
    abs_time = int(time_ary[0])*60+int(time_ary[1])+float(time_ary[2])/100
    finish_time_test.append(abs_time)

finish_time_train = np.asarray(finish_time_train)
finish_time_test = np.asarray(finish_time_test)

scaler = StandardScaler()
X_train = train[['actual_weight', 'declared_horse_weight', 'draw', 'win_odds', 'jockey_ave_rank', 'trainer_ave_rank', 'recent_ave_rank', 'race_distance']].values
y_train = finish_time_train
X_train_scaled = scaler.fit_transform(X_train)
y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

X_test = test[['actual_weight', 'declared_horse_weight', 'draw', 'win_odds', 'jockey_ave_rank', 'trainer_ave_rank', 'recent_ave_rank', 'race_distance']].values
y_test = finish_time_test
X_test_scaled = scaler.fit_transform(X_test)
y_test_scaled = scaler.fit_transform(y_test.reshape(-1, 1)).ravel()

print("Performing SVR...")
time_svr = time.time()
svr_model = SVR(kernel='rbf', C=1, epsilon=0.01, gamma=1e-4)
svr_model.fit(X_train, y_train)
svr_pred = svr_model.predict(X_test)
test['svr_predict_finish_time'] = svr_pred
error_svr = sqrt(mean_squared_error(y_test, svr_pred))
print("root mean squared error of svr model: ", error_svr)
#
print("After Normalization...")
svr_model.fit(X_train_scaled, y_train_scaled)
svr_pred_scaled = svr_model.predict(X_test_scaled)
test['svr_predict_scaled_finish_time'] = svr_pred_scaled
error_svr_scaled = sqrt(mean_squared_error(y_test_scaled, svr_pred_scaled))
print("root mean squared error of svr model after normalization: ", error_svr_scaled)
print("Running time of logistic regression: %.2fs" % (time.time()-time_svr))
print("Done")

print("Performing Gradient Boosting Regressor...")
time_gbr = time.time()
gbr_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls')
gbr_model.fit(X_train, y_train)
gbr_pred = gbr_model.predict(X_test)
test['gbr_predict_finish_time'] = gbr_pred
error_gbr = sqrt(mean_squared_error(y_test, gbr_pred))
print("root mean squared error of gbr model: ", error_gbr)
#
print("After Normalization...")
gbr_model.fit(X_train_scaled, y_train_scaled)
gbr_pred_scaled = gbr_model.predict(X_test_scaled)
test['gbr_predict_scaled_finish_time'] = gbr_pred_scaled
error_gbr_scaled = sqrt(mean_squared_error(y_test_scaled, gbr_pred_scaled))
print("root mean squared error of gbr model after normalization: ", error_gbr_scaled)
print("Running time of gbr: %.2fs" % (time.time()-time_gbr))
print("Done")

print("Evaluating models...")

svr_sum_top1 = 0
svr_sum_top3 = 0
svr_rank = []
gbr_sum_top1 = 0
gbr_sum_top3 = 0
gbr_rank = []

svr_sum_top1_scaled = 0
svr_sum_top3_scaled = 0
svr_rank_scaled = []
gbr_sum_top1_scaled = 0
gbr_sum_top3_scaled = 0
gbr_rank_scaled = []

for data in test.race_id.values:
    svr_fin_time = test.loc[test['race_id']==data].svr_predict_finish_time.values
    gbr_fin_time = test.loc[test['race_id']==data].gbr_predict_finish_time.values
    svr_fin_time_scaled = test.loc[test['race_id']==data].svr_predict_scaled_finish_time.values
    gbr_fin_time_scaled = test.loc[test['race_id']==data].gbr_predict_scaled_finish_time.values
    
    actual_fin_pos = test.loc[test['race_id']==data].finishing_position.values
    svr_min_time_idx = svr_fin_time.tolist().index(min(svr_fin_time))
    gbr_min_time_idx = gbr_fin_time.tolist().index(min(gbr_fin_time))
    svr_min_time_idx_scaled = min(enumerate(svr_fin_time_scaled.tolist()), key=itemgetter(1))[0] 
    gbr_min_time_idx_scaled = min(enumerate(gbr_fin_time_scaled.tolist()), key=itemgetter(1))[0]
    
    svr_rank.append(actual_fin_pos[svr_min_time_idx])
    gbr_rank.append(actual_fin_pos[gbr_min_time_idx])
    svr_rank_scaled.append(actual_fin_pos[svr_min_time_idx_scaled])
    gbr_rank_scaled.append(actual_fin_pos[gbr_min_time_idx_scaled])
    
    if(svr_min_time_idx+1 == 1):
        svr_sum_top1 += 1
    if(svr_min_time_idx+1 <= 3):
        svr_sum_top3 +=1
    if(gbr_min_time_idx+1 == 1):
        gbr_sum_top1 += 1
    if(svr_min_time_idx+1 <= 3):
        gbr_sum_top3 +=1
        
    if(svr_min_time_idx_scaled+1 == 1):
        svr_sum_top1_scaled += 1
    if(svr_min_time_idx_scaled+1 <= 3):
        svr_sum_top3_scaled +=1
    if(gbr_min_time_idx_scaled+1 == 1):
        gbr_sum_top1_scaled += 1
    if(svr_min_time_idx_scaled+1 <= 3):
        gbr_sum_top3_scaled +=1

svr_top1 = svr_sum_top1/len(test.race_id.values)
svr_top3 = svr_sum_top3/len(test.race_id.values)
svr_average_rank = (sum(svr_rank)/len(svr_rank))

gbr_top1 = gbr_sum_top1/len(test.race_id.values)
gbr_top3 = gbr_sum_top3/len(test.race_id.values)
gbr_average_rank = (sum(gbr_rank)/len(gbr_rank))

svr_top1_scaled = svr_sum_top1_scaled/len(test.race_id.values)
svr_top3_scaled = svr_sum_top3_scaled/len(test.race_id.values)
svr_average_rank_scaled = (sum(svr_rank_scaled)/len(svr_rank_scaled))

gbr_top1_scaled = gbr_sum_top1_scaled/len(test.race_id.values)
gbr_top3_scaled = gbr_sum_top3_scaled/len(test.race_id.values)
gbr_average_rank_scaled = (sum(gbr_rank_scaled)/len(gbr_rank_scaled))

print("Prediction Evaluation:")
print("----Without normalization----")
print("Support Vector Regression:")
print("Top_1: %.2f, Top_3: %.2f, Average_rank: %.2f"%(svr_top1, svr_top3, svr_average_rank))
print("Gradient Boosting Regressor:")
print("Top_1: %.2f, Top_3: %.2f, Average_rank: %.2f"%(gbr_top1, gbr_top3, gbr_average_rank))
print("----With normalization----")
print("Support Vector Regression:")
print("Top_1: %.2f, Top_3: %.2f, Average_rank: %.2f"%(svr_top1_scaled, svr_top3_scaled, svr_average_rank_scaled))
print("Gradient Boosting Regressor:")
print("Top_1: %.2f, Top_3: %.2f, Average_rank: %.2f"%(gbr_top1_scaled, gbr_top3_scaled, gbr_average_rank_scaled))
