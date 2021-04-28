from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from dataloaders import fullDataLoader, byTarget
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import csv
# from scipy import 



def accuracy(labels, outputs, percent=.10):
    llabels = labels
    loutputs = outputs
    correct = 0
    total = 0
    # print(llabels)
    for i in range(len(llabels)):
        if llabels[i] * (1 - percent) <= loutputs[i] and llabels[i] * (1 + percent) >= loutputs[i]:
            correct +=1
        total += 1

    return correct / total    

if __name__ == "__main__":
    # model = RandomForestClassifier(n_estimators=100, verbose=False)
    # print('loading data')
    # data, _ = fullDataLoader(batch=1, target='thresh')
    # with open('augmentcrisprsql.csv') as f:
    #     df = pd.DataFrame(byTarget(list(csv.DictReader(f)))[2])


    # x = []
    # y = []
    # for row in data[0]:
    #     l = row[0][0].flatten().tolist()
    #     l.extend(row[0][1].flatten().tolist())
    #     x.append(l)
    #     y.append(row[1].flatten()[0])

    # xt=[]
    # yt = []
    # for row in data[2]:
    #     l = row[0][0].flatten().tolist()
    #     l.extend(row[0][1].flatten().tolist())
    #     xt.append(l)
    #     yt.append(row[1].flatten()[0])
    # model.fit(x, y)
    # # print(model.predict(xt))
    # total = 0
    # cor = 0

    # for yhat, y in zip(y,model.predict(x)):
    #     if float(yhat) == float(y):
    #         cor+=1
    #     total+=1
    # print('acc train', cor/total)
    # total = 0
    # cor = 0
    # for yhat, y in zip(yt,model.predict(xt)):
    #     if float(yhat) == float(y):
    #         cor+=1
    #     total+=1

   

    # print('roc', roc_auc_score(yt,  model.predict(xt)))
    # print('acc', cor/total)

    # df['threshhold_random_rf'] = pd.Series(model.predict(xt))


    # ranked
    
    model = RandomForestRegressor(n_estimators=20, verbose=False)
    print('loading data')
    data, _ = fullDataLoader(batch=1)

    x = []
    y = []
    for row in data[0][:200]:
        l = row[0][0].flatten().tolist()
        l.extend(row[0][1].flatten().tolist())
        x.append(l)
        y.append(row[1].flatten().item())

    xt=[]
    yt = []
    # for row in data[2]:
    #     l = row[0][0].flatten().tolist()
    #     l.extend(row[0][1].flatten().tolist())
    #     xt.append(l)
    #     yt.append(row[1].flatten().item())
    model.fit(x, y)
    
    total = len(y)
    cor = accuracy(y, model.predict(x)) * total
    print('train pearsonr, p-value: ', pearsonr(y, model.predict(x)))
    print('spearmanr:', spearmanr(y, model.predict(x)))
    print('acc train', cor/total)
    
    # print(model.predict(xt))
    total = len(y)
    cor = accuracy(y, model.predict(x)) * total

    # print(zip(y, model.predict(x)))
    pearson = [
        {'yhat': p1, 'ypred': p} for p1, p in zip(y, model.predict(x))
    ]
    df = pd.DataFrame(pearson)
    print(df)

    # sns.scatterplot(x='p^', y='p', data=df)
    # plt.tight_layout()
    # plt.show()

    print('test pearsonr, p-value: ', pearsonr(y, model.predict(x)))
    print('acc', cor/total)
    # df['ranked_random_rf'] = pd.Series(model.predict(x))
    df.to_csv('randomforestcrisprsql_ONLYTEST_rank.csv')

    
    # regular data



    # model = RandomForestRegressor(n_estimators=1000, verbose=False)
    # print('loading data')
    # data, _ = fullDataLoader(batch=1, target='regular')[:200]

    # x = []
    # y = []
    # for row in data[0]:
    #     l = row[0][0].flatten().tolist()
    #     l.extend(row[0][1].flatten().tolist())
    #     x.append(l)
    #     y.append(row[1].flatten().item())

    # xt=[]
    # yt = []
    # # for row in data[2]:
    # #     l = row[0][0].flatten().tolist()
    # #     l.extend(row[0][1].flatten().tolist())
    # #     xt.append(l)
    # #     yt.append(row[1].flatten().item())
    # model.fit(x, y)
    
    # total = len(y)
    # pred = model.predict(x)
    # cor = accuracy(y, pred) * total
    # print('train pearsonr, p-value: ', pearsonr(y, pred))
    # print('acc train', cor/total)
    
    # # print(model.predict(xt))
    # pred1 =  model.predict(x)
    # total = len(y)
    # cor = accuracy(y, pred1) * total

    # pearson = [
    #     {'yhat': p1, 'ypred': p} for p1, p in zip(y, model.predict(x))
    # ]
    # df = pd.DataFrame(pearson)

    # # sns.scatterplot(x='p^', y='p', data=df_)
    # # plt.tight_layout()
    # # plt.show()

    # # print('roc', roc_auc_score(y,  model.predict(x)))
    # print('test pearsonr, p-value: ', pearsonr(y, model.predict(x)))
    # print('spearmanr:', spearmanr(y, model.predict(x)))
    # print('acc', cor/total)
    # df['regular_random_rf'] = pd.Series(pred1)

    # df.to_csv('randomforestcrisprsql_ONLYTEST_reg.csv')




