from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from naive_bayes import MultinomialNB
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np
import pandas as pd
import time


train = pd.read_csv('./data/training.csv')
test = pd.read_csv('./data/testing.csv')


X_train = train[['recent_ave_rank', 'draw', 'race_distance', 'trainer_ave_rank', 'jockey_ave_rank', 'actual_weight', 'win_odds']].values
y1_train = train[['HorseWin']].values.ravel()
y2_train = train[['HorseRankTop3']].values.ravel()
y3_train = train[['HorseRankTop50Percent']].values.ravel()

X_test = test[['recent_ave_rank', 'draw', 'race_distance', 'trainer_ave_rank', 'jockey_ave_rank', 'actual_weight', 'win_odds']].values
y1_test = test[['HorseWin']].values.ravel()
y2_test = test[['HorseRankTop3']].values.ravel()
y3_test = test[['HorseRankTop50Percent']].values.ravel()


kfold = KFold(n_splits=10)

#Logistic Regression#
print("Performing logistic regression...")
time_lr = time.time()
lr_model = LogisticRegression()
lr_model.fit(X_train, y1_train)
y_pred_lr_1 = lr_model.predict(X_test)
#results = cross_val_score(lr_model, X_train, y1_train, cv=kfold, scoring='accuracy')
#print(results)

lr_model.fit(X_train, y2_train)
y_pred_lr_2 = lr_model.predict(X_test)
#results = cross_val_score(lr_model, X_train, y2_train, cv=kfold, scoring='accuracy')
#print(results)

lr_model.fit(X_train, y3_train)
y_pred_lr_3 = lr_model.predict(X_test)
#results = cross_val_score(lr_model, X_train, y3_train, cv=kfold, scoring='accuracy')
#print(results)

print("Running time of logistic regression: %.2fs" % (time.time()-time_lr))
print("Done")

#Naive Bayes(MultinomialNB)#
print("Performing naive bayes...")
time_nb = time.time()
nb_model = MultinomialNB()
nb_model.fit(X_train, y1_train)
y_pred_nb_1 = nb_model.predict(X_test)

nb_model.fit(X_train, y2_train)
y_pred_nb_2 = nb_model.predict(X_test)

nb_model.fit(X_train, y3_train)
y_pred_nb_3 = nb_model.predict(X_test)

print("Running time of naive bayes: %.2fs" % (time.time()-time_nb))
print("Done")

#Support Vector Machine#
print("Performing SVM...")
time_svm = time.time()
svm_model = svm.SVC(C=0.5, kernel='rbf', random_state=3320)
svm_model.fit(X_train, y1_train)
y_pred_svm_1 = svm_model.predict(X_test)

svm_model.fit(X_train, y2_train)
y_pred_svm_2 = svm_model.predict(X_test)

svm_model.fit(X_train, y3_train)
y_pred_svm_3 = svm_model.predict(X_test)

print("Running time of svm: %.2fs" % (time.time()-time_svm))
print("Done")

#Random Forest#
print("Performing random forest...")
time_rf = time.time()
rf_model = RandomForestClassifier(n_estimators=60, max_features='sqrt', random_state=3320)
rf_model.fit(X_train, y1_train)
y_pred_rf_1 = rf_model.predict(X_test)

rf_model.fit(X_train, y2_train)
y_pred_rf_2 = rf_model.predict(X_test)

rf_model.fit(X_train, y3_train)
y_pred_rf_3 = rf_model.predict(X_test)

print("Running time of random forest: %.2fs" % (time.time()-time_rf))
print("Done")

#Prediction evaluation#
print("Evaluatin prediction...")
print("Logistic Regression:")
print("         HorseWin HorseRankTop3 HorseRankTop50Percent")
p_score_lr_1 = precision_score(y1_test, y_pred_lr_1)
p_score_lr_2 = precision_score(y2_test, y_pred_lr_2)
p_score_lr_3 = precision_score(y3_test, y_pred_lr_3)
print("P score: %-8.2f %-13.2f %-21.2f"%(p_score_lr_1, p_score_lr_2, p_score_lr_3))
r_score_lr_1 = recall_score(y1_test, y_pred_lr_1)
r_score_lr_2 = recall_score(y2_test, y_pred_lr_2)
r_score_lr_3 = recall_score(y3_test, y_pred_lr_3)
print("R score: %-8.2f %-13.2f %-21.2f"%(r_score_lr_1, r_score_lr_2, r_score_lr_3))

print("Navie Bayes(MultinomialNB):")
print("         HorseWin HorseRankTop3 HorseRankTop50Percent")
p_score_nb_1 = precision_score(y1_test, y_pred_nb_1)
p_score_nb_2 = precision_score(y2_test, y_pred_nb_2)
p_score_nb_3 = precision_score(y3_test, y_pred_nb_3)
print("P score: %-8.2f %-13.2f %-21.2f"%(p_score_nb_1, p_score_nb_2, p_score_nb_3))
r_score_nb_1 = recall_score(y1_test, y_pred_nb_1)
r_score_nb_2 = recall_score(y2_test, y_pred_nb_2)
r_score_nb_3 = recall_score(y3_test, y_pred_nb_3)
print("R score: %-8.2f %-13.2f %-21.2f"%(r_score_nb_1, r_score_nb_2, r_score_nb_3))

print("Support Vector Machine:")
print("         HorseWin HorseRankTop3 HorseRankTop50Percent")
p_score_svm_1 = precision_score(y1_test, y_pred_svm_1)
p_score_svm_2 = precision_score(y2_test, y_pred_svm_2)
p_score_svm_3 = precision_score(y3_test, y_pred_svm_3)
print("P score: %-8.2f %-13.2f %-21.2f"%(p_score_svm_1, p_score_svm_2, p_score_svm_3))
r_score_svm_1 = recall_score(y1_test, y_pred_svm_1)
r_score_svm_2 = recall_score(y2_test, y_pred_svm_2)
r_score_svm_3 = recall_score(y3_test, y_pred_svm_3)
print("R score: %-8.2f %-13.2f %-21.2f"%(r_score_svm_1, r_score_svm_2, r_score_svm_3))

print("Random Forest:")
print("         HorseWin HorseRankTop3 HorseRankTop50Percent")
p_score_rf_1 = precision_score(y1_test, y_pred_rf_1)
p_score_rf_2 = precision_score(y2_test, y_pred_rf_2)
p_score_rf_3 = precision_score(y3_test, y_pred_rf_3)
print("P score: %-8.2f %-13.2f %-21.2f"%(p_score_rf_1, p_score_rf_2, p_score_rf_3))
r_score_rf_1 = recall_score(y1_test, y_pred_rf_1)
r_score_rf_2 = recall_score(y2_test, y_pred_rf_2)
r_score_rf_3 = recall_score(y3_test, y_pred_rf_3)
print("R score: %-8.2f %-13.2f %-21.2f"%(r_score_rf_1, r_score_rf_2, r_score_rf_3))



print("Writing results to csv...")
num_pred = y_pred_lr_1.size
lr_file = open("./predictions/lr_predictions.csv",'w+')
lr_file.write('RaceID,HorseID,HorseWin,HorseRankTop3,HorseRankTop50Percent\n')
nb_file = open("./predictions/nb_predictions.csv",'w+')
nb_file.write('RaceID,HorseID,HorseWin,HorseRankTop3,HorseRankTop50Percent\n')
svm_file = open("./predictions/svm_predictions.csv",'w+')
svm_file.write('RaceID,HorseID,HorseWin,HorseRankTop3,HorseRankTop50Percent\n')
rf_file = open("./predictions/rf_predictions.csv",'w+')
rf_file.write('RaceID,HorseID,HorseWin,HorseRankTop3,HorseRankTop50Percent\n')

race_ids = test.race_id.values
horse_ids = test.horse_id.values
for i in range(num_pred):
#    print(race_ids[i], horse_ids[i], y_pred_lr_1[i], y_pred_lr_2[i], y_pred_lr_3[i])
    lr_file.write('%s,%s,%d,%d,%d\n'%(race_ids[i], horse_ids[i], y_pred_lr_1[i], y_pred_lr_2[i], y_pred_lr_3[i]))
    nb_file.write('%s,%s,%d,%d,%d\n'%(horse_ids[i], race_ids[i], y_pred_nb_1[i], y_pred_nb_2[i], y_pred_nb_3[i]))
    svm_file.write('%s,%s,%d,%d,%d\n'%(race_ids[i], horse_ids[i], y_pred_svm_1[i], y_pred_svm_2[i], y_pred_svm_3[i]))
    rf_file.write('%s,%s,%d,%d,%d\n'%(race_ids[i], horse_ids[i], y_pred_rf_1[i], y_pred_rf_2[i], y_pred_rf_3[i]))
    
print("Done")
