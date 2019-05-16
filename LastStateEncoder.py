from sklearn.base import TransformerMixin
import pandas as pd
import time
from datetime import datetime, timedelta

class LastStateEncoder(TransformerMixin):
    
    def __init__(self, case_id_col, timestamp_col, cat_cols, numeric_cols, fillna=True):
        self.case_id_col = case_id_col
        self.timestamp_col = timestamp_col
        self.cat_cols = cat_cols
        self.numeric_cols = numeric_cols
        self.fillna = fillna
        self.columns = None

    def throughput(self,x):    
        i=x.index[0]
        t1=x['Start Timestamp'][i]
        t= time.strptime(t1, "%Y/%m/%d %H:%M:%S.%f")
        i+=1
        sys=0  # delay by system
        usr=0  # delay by user
        system=True
        while i < x.index.max():
            #print('before ', i)
            if x['Activity'][i]=='O_Sent (mail and online)' or x['Activity'][i]=='O_Sent (online only)':
                t2=x['Complete Timestamp'][i]
                t2 = time.strptime(t2, "%Y/%m/%d %H:%M:%S.%f")
                s=datetime.fromtimestamp(time.mktime(t2))-datetime.fromtimestamp(time.mktime(t))
                sys+=86400 * s.days + s.seconds
                system=False
                t=t2
                #print('send ',sys)

            if x['Activity'][i]=='A_Validating' :
                t2=x['Start Timestamp'][i]
                t2 = time.strptime(t2, "%Y/%m/%d %H:%M:%S.%f")
                u=datetime.fromtimestamp(time.mktime(t2))-datetime.fromtimestamp(time.mktime(t))
                usr+=86400 * u.days + u.seconds
                system=True
                t=t2
                #print('val :',usr)

            i+=1
            #print(i)
            if i==x.index.max():
                if system:
                    t2=x['Complete Timestamp'][i]
                    t2 = time.strptime(t2, "%Y/%m/%d %H:%M:%S.%f")
                    s=datetime.fromtimestamp(time.mktime(t2))-datetime.fromtimestamp(time.mktime(t))
                    sys+=86400 * s.days + s.seconds
                    #print('final sys  ',sys)
                else:
                    t2=x['Complete Timestamp'][i]
                    t2 = time.strptime(t2, "%Y/%m/%d %H:%M:%S.%f")
                    u=datetime.fromtimestamp(time.mktime(t2))-datetime.fromtimestamp(time.mktime(t))
                    usr+=86400 * u.days + u.seconds
                    #print('fainal usr ',usr)
        return sys,usr
    
    def fit(self, X, y=None):
        return self
    def sort_and_calculate(self,x):
        sys,usr=self.throughput(x)
        th_col=['sys','usr']
        x=x.tail(1).drop(self.timestamp_col, axis=1)
        #x=pd.concat([x,th_col],axis=1)
        x['sys']=sys
        x['usr']=usr
        return x

    def transform(self, data, y=None):

        # taking throughput
        
        
        # reshape: each activity will be separate column
        data_cat = pd.get_dummies(data[self.cat_cols])
        data = pd.concat([data[[self.case_id_col]+['Activity']+self.timestamp_col+self.numeric_cols], data_cat], axis=1)
        
        # aggregate activities by case
        grouped = data.groupby(self.case_id_col)
        
        # extract values from last event
        #data = grouped.apply(lambda x: x.sort_values(by=self.timestamp_col, ascending=True).tail(1)).drop(self.timestamp_col, axis=1)
        data = grouped.apply(lambda x: self.sort_and_calculate(x).drop(['Activity'],axis=1))

        # fill missing values with 0-s
        if self.fillna:
            data.fillna(0, inplace=True)
        
        # add missing columns if necessary
        if self.columns is None:
            self.columns = data.columns
        else:
            missing_cols = [col for col in self.columns if col not in data.columns]
            for col in missing_cols:
                data[col] = 0
            data = data[self.columns]
            
        return data