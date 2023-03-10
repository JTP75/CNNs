import pandas as pd

class dataframe_t:
    def __init__(self,df: pd.DataFrame):
        self.df = df
        self.labels = []
        self.dummies = []
        self.df_num = None
    def num(self,cols: list or tuple) -> pd.DataFrame:
        if len(self.labels)==0:
            for label in cols:
                self.__num__(label)
        return self.df_num
    def __num__(self,label) -> None:
        self.labels.append(self.df[label])
        self.dummies.append(pd.get_dummies(self.df[label]))
        self.df_num = pd.concat((self.df,self.dummies[-1]),axis=1).reindex(self.df.index())
        self.df_num.drop(label)
        return self.df_num