import pandas as pd
from sklearn.model_selection import train_test_split

emotions = ['anger','fear','joy','sadness']
seed = 41

for i in emotions:
    df1 = pd.read_csv(f'raw_data\\{i}\\{i}-ratings-0to1.train.txt', sep='\t', header=None, names=['ID', 'Tweet', 'Emotion', 'Intensity'])
    df2 = pd.read_csv(f'raw_data\\{i}\\{i}-ratings-0to1.test.gold.txt', sep='\t', header=None, names=['ID', 'Tweet', 'Emotion', 'Intensity'])
    df3 = pd.read_csv(f'raw_data\\{i}\\{i}-ratings-0to1.dev.gold.txt', sep='\t', header=None, names=['ID', 'Tweet', 'Emotion', 'Intensity'])

    df = pd.concat([df1,df2,df3], ignore_index=True)

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    train_df, remaining_df = train_test_split(df, test_size=0.2,random_state=seed)
    val_df,test_df = train_test_split(remaining_df,test_size=0.5,random_state=seed)

    train_df.to_csv(f'tuned_data\\{i}\\{i}_training_data.csv',sep='\t',index=False)
    val_df.to_csv(f'tuned_data\\{i}\\{i}_validation_data.csv',sep='\t',index=False)
    test_df.to_csv(f'tuned_data\\{i}\\{i}_test_data.csv',sep='\t',index=False)