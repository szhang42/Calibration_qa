import numpy as np
import json, string
import xgboost as xgb
import sys, os
from sklearn.metrics import roc_auc_score, roc_curve, auc
import pickle
from sklearn.decomposition import PCA

from sklearn.metrics import accuracy_score
import random
inputs = [] # input
labels = [] # label


count = 0
add_negative_flag = False
# with open('dpr_hidden_features_dev_normalized_all_passages.json', 'rb') as f:
with open('dpr_topone_hidden_passage@100_data@dev.json', 'rb') as f:
    inputs = []
    labels = []
    while True:
        count += 1
        data = f.readline()
        if count % 100 == 0:
            print('read one line', count)
        if len(data) <= 10:
            break
        try:
            data = json.loads(data)
        except:
            print(count, 'error')
            continue

        thres = 1e-2 # 1e-4 # 2
        for index in range(len(data)):
            if data[index][1] * data[index][0] < thres:
                continue
            if data[index][-1] == 1:
                # if data[index][1] >= thres and data[index][0] >= thres: # random.randint(0, 5) <= 4:
                #x1 = np.array(data[index][2+768:-1])
                #x2 = np.array(data[index][2:768+2])
                #inputs.append([data[index][0] * data[index][1]] + data[index][:2] + [(x1 * x2).sum() / np.linalg.norm(x1) / np.linalg.norm(x2), np.linalg.norm(x1 - x2)])
                inputs.append([data[index][0] * data[index][1]] + data[index][:-1])
                # inputs.append(data[index][:-1])
                labels.append(data[index][-1])
                '''
                else:
                    inputs.append([data[index][0] * data[index][1]] + data[index][:2])
                    # inputs.append(data[index][2:-1])
                    labels.append(0) # data[index][-1])
                '''
            else:
                # inputs.append(data[index][:-1])
                #x1 = np.array(data[index][2+768:-1])
                #x2 = np.array(data[index][2:768+2])
                #inputs.append([data[index][0] * data[index][1]] + data[index][:2] + [(x1 * x2).sum() / np.linalg.norm(x1) / np.linalg.norm(x2), np.linalg.norm(x1 - x2)])
                                
                inputs.append([data[index][0] * data[index][1]] + data[index][:-1])
                labels.append(data[index][-1])
        # inputs += [[item[0] * item[1]] + item[:2] for item in data]
        # inputs += [[item[0] * item[1]] + item[:2] + item[2+768:-1] for item in data]
        # labels += [item[-1] for item in data]
        # if count >= 220:
        #     break
        '''
        for ques_index in range(len(data)):
            true_answer = data[ques_index][-1][0].lower()
            answers = []
            answer_hiddens = []
            logits = []
            for context_index in range(len(data[ques_index][1])):
                answer = data[ques_index][1][context_index][0].lower()
                answer_hiddens.append(data[ques_index][1][context_index][-1])
                logits.append(data[ques_index][1][context_index][1:3])
                _hidden = data[ques_index][1][context_index][3]
                # print(type(_hidden))
                if context_index == 0:
                    hidden = data[ques_index][1][context_index][3]
                # _hidden = data[ques_index][1][context_index][3]
                # if isinstance(_hidden, list):
                #     hidden = data[ques_index][1][context_index][3]
                
                answers.append(answer)
            print(true_answer);input()
            for answer_index in range(len(answers)):
                inputs.append(logits[answer_index] + hidden + answer_hiddens[answer_index])
                if true_answer == answers[answer_index]:
                    labels.append(1)
                else:
                    labels.append(0)
        if count > 200:
            break
        '''

x = np.array(inputs)
label = np.array(labels, dtype=int)
print(x.shape, label.shape)
print(np.sum(label))
interval = x.shape[0] // 50 * 49
acc = 0.
auc = 0.
k_times = 1
for _ in range(k_times):
    # shuffle data
    index = np.random.permutation(label.shape[0])
     
    shuffled_x = x[index]; shuffled_label = label[index]
    
    train_data = shuffled_x[:interval, :]; train_label = shuffled_label[:interval]
    test_data = shuffled_x[interval:, :]; test_label = shuffled_label[interval:]

    colsample = 1.
    depth = 25
    iteration = 50
    xg_cls = xgb.XGBClassifier(objective='binary:logistic', colsample_bylevel=colsample, colsample_bynode=colsample, colsample_bytree=colsample, max_depth=depth, n_estimators=iteration, learning_rate=.3)
    xg_cls.fit(train_data[:, :3], train_label, eval_metric=['auc', 'logloss'], eval_set=[(train_data[:1000, :3], train_label[:1000]), (test_data[:, :3], test_label)], verbose=True)
    with open('xgboost_passage@100_thres@0.01_data@dev_depth@'+str(depth)+'_dim@3', 'wb') as f:
        pickle.dump(xg_cls, f)
    xg_cls = xgb.XGBClassifier(objective='binary:logistic', colsample_bylevel=colsample, colsample_bynode=colsample, colsample_bytree=colsample, max_depth=depth, n_estimators=iteration, learning_rate=.3)
    xg_cls.fit(train_data, train_label, eval_metric=['auc', 'logloss'], eval_set=[(train_data[:1000], train_label[:1000]), (test_data, test_label)], verbose=True)
    with open('xgboost_passage@100_thres@0.01_data@dev_depth@'+str(depth)+'_dim@full', 'wb') as f:
        pickle.dump(xg_cls, f)    
   

    preds = xg_cls.predict_proba(test_data)
    auc += roc_auc_score(test_label, preds[:, 1])
    preds = xg_cls.predict(test_data)
    print(sum(preds))
    acc += accuracy_score(test_label, xg_cls.predict(test_data)) # np.where(test_label == np.ones_like(test_label), np.equal(preds, test_label), np.zeros_like(test_label)).sum() * 1.0 / test_label.sum()
    
    print(acc / (_+1), auc / (_+1))
print(acc / k_times, auc /k_times)



