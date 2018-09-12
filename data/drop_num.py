import pandas as pd

def drop_num(f):
    df = pd.read_csv(f)
    num = set()
    for i in range(len(df)):
        sentence = df.iloc[i].sentence
        index = df[df.sentence == sentence]
        if len(index) > 2:
            num.add(index.index[-1])
            num.add(index.index[-2])
        elif len(index) == 2:
            num.add(index.index[-1])
    list_num = [i for i in num]
    list_num.sort(reverse=True)
    for i in list_num:
        df.drop(i,inplace=True)

def compare_sentence(f1,f2,f3):
    import ast
    from collections import Counter
    f4 = pd.DataFrame(columns=['sentence','entity','score'])
    f5 = pd.read_csv(f1, index_col=0)
    temp = [f1,f2,f3]
    f1 = pd.read_csv(f1, index_col=0)
    f1.score = f1.score.apply(lambda s: list(ast.literal_eval(s)))
    f2 = pd.read_csv(f2, index_col=0)
    f2.score = f2.score.apply(lambda s: list(ast.literal_eval(s)))
    f3 = pd.read_csv(f3, index_col=0)
    f3.score = f3.score.apply(lambda s: list(ast.literal_eval(s)))
    num = []
    for i in range(len(f1)):
        score1 = int(f1.iloc[i].score[0])
        score2 = int(f2.iloc[i].score[0])
        score3 = int(f3.iloc[i].score[0])
        Temp = [score1,score2,score3]
        Temp = Counter(Temp)
        if len(Temp) >2:
            f4 = f4.append(f1.iloc[i])
            f5.drop(index=i,inplace=True)
            num.append(i)
        else:
            score = Temp.most_common(1)[0][0]
            f5.iloc[i].score = score
    print(num)










