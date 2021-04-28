import pandas as pd
import csv
import numpy as np
with open('crisprsql.csv') as f:
    data = list(csv.DictReader(f))
    df = pd.DataFrame(data)
    # pd.to_numeric(df.loc[:, 'cleavage_freq'],  errors='coerce')
    print(df['cleavage_freq'].head())
    df['cleavage_freq'] = df['cleavage_freq'].apply(pd.to_numeric, args=('coerce',))
    # df.dropna(axis=0, subset=['cleavage_freq'])

    
    df =  df[df['cleavage_freq'].notna()]
    print(type(df.at[0, 'cleavage_freq']))
    df['ranked_cleavage_freq'] = np.nan
    df['threshhold_cleavage_freq'] = np.nan
    ranks = list()
    for index, line in df.iterrows():
        if type(line['cleavage_freq']) == str:
            print(line['cleavage_freq'] )
        if line['cleavage_freq'] != '' and float(line['cleavage_freq']) >= 0:
            ranks.append(float(line['cleavage_freq']))
    ranks.sort()
    for index, line in df.iterrows():
        if line['cleavage_freq'] != '' and float(line['cleavage_freq']) >= 0:
            df.at[index, "threshhold_cleavage_freq"] = 1 if line['cleavage_freq'] >= 0.01 else 0
            df.at[index, 'ranked_cleavage_freq'] = ranks.index(float(line['cleavage_freq'])) / len(ranks)
    print(df.head())
    
    df.to_csv('augmentcrisprsql.csv')
