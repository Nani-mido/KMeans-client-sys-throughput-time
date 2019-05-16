from FrequencyEncoder import FrequencyEncoder
from LastStateEncoder import LastStateEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import time
from datetime import datetime, timedelta
import numpy as np

class ClusteringPredictiveModel:
    
    def __init__(self, case_id_col, event_col, label_col, timestamp_col, cat_cols, numeric_cols, n_clusters,n_sub_family, n_estimators, random_state=22, fillna=True, pos_label="A_Pending"):
        
        # columns
        self.case_id_col = case_id_col
        self.label_col = label_col
        self.pos_label = pos_label
        self.avg_sys=0
        self.avg_usr=0
        self.n_clusters = n_clusters
        
        self.freq_encoder = FrequencyEncoder(case_id_col, event_col)
        self.data_encoder = LastStateEncoder(case_id_col, timestamp_col, cat_cols, numeric_cols, fillna)
        self.clustering = KMeans(n_clusters,  random_state=random_state)
        self.sub_clustering = KMeans(n_sub_family,  random_state=random_state)
        self.clss = [RandomForestClassifier(n_estimators=n_estimators, random_state=random_state) for _ in range(n_clusters*n_sub_family)]
        
        self.avg_time_train=np.zeros(n_clusters)
        self.avg_all_train=0
        self.avg_time_test=np.zeros(n_clusters)
        self.avg_all_test=0
        self.n_sub_family=n_sub_family
        self.sys=np.zeros(n_clusters*n_sub_family)
        self.usr=np.zeros(n_clusters*n_sub_family)
        self.tot=np.zeros(n_clusters*n_sub_family)

    def throughput(self,x):   #for one case
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
        
        return np.array([sys,usr])

    def execution_time(self,x): #for one case
        t=x['Start Timestamp'][x.index.min()]
        t = time.strptime(t, "%Y/%m/%d %H:%M:%S.%f")
        t2=x['Complete Timestamp'][x.index.max()]
        t2 = time.strptime(t2, "%Y/%m/%d %H:%M:%S.%f")
        exe=datetime.fromtimestamp(time.mktime(t2))-datetime.fromtimestamp(time.mktime(t))
        return (86400 * exe.days + exe.seconds)

    def average_in_cluster(self,x):
        return np.mean(x['sys']+x['usr'])

    def sub_family(self,x):# for one case
        if x['sys'][0]>self.avg_sys and x['usr'][0]>self.avg_usr:   #ss
            return 0
        elif x['sys'][0]>self.avg_sys and x['usr'][0]<self.avg_usr: #sf
            return 1
        elif x['sys'][0]<self.avg_sys and x['usr'][0]>self.avg_usr: #fs
            return 2
        elif x['sys'][0]<self.avg_sys and x['usr'][0]<self.avg_usr: #ff
            return 3 


    def fit(self, X, y=None):
        
        # encode events as frequencies
        data_freqs = self.freq_encoder.fit_transform(X)
        
        # cluster traces according to event frequencies 
        cluster_assignments = self.clustering.fit_predict(data_freqs)
        
        #avg_grouping=X.groupby(self.case_id_col)
        #ex=avg_grouping.apply(lambda x: execution_time(x))
        #self.avg=ex.mean()
        
        #avg_grouping1=X.groupby(['Case ID'])
        #avg1=avg_grouping1.apply(lambda x: self.throughput(x))
        #self.avg_sys=np.mean([avg1[p][0] for p in range(len(avg1))])
        #self.avg_usr=np.mean([avg1[p][1] for p in range(len(avg1))])

        
        # train classifier for each cluster
        shift=0
        for cl in range(self.n_clusters):

            #print('Distribution of cluster ',cl,':', end='')
            cases = data_freqs[cluster_assignments == cl].index
            tmp = X[X[self.case_id_col].isin(cases)]
            tmp = self.data_encoder.transform(tmp)
            self.avg_time_train[cl]=self.average_in_cluster(tmp)
            sub_fammily_assignments=self.sub_clustering.fit_predict(tmp[['usr']])
            mean=0
            for s_cl in range (self.n_sub_family):
                print('cluster ',s_cl+shift,':', end='')
                s_cases=tmp[sub_fammily_assignments==s_cl]['Case ID']
                tempo=tmp[tmp[self.case_id_col].isin(s_cases)]
                self.clss[s_cl+shift].fit(tempo.drop([self.case_id_col, self.label_col,'sys','usr'], axis=1), tempo[self.label_col])
                avg=int(round(self.average_in_cluster(tempo)/86400))
                avg_usr=int(round(np.mean(tempo['usr'])/86400))
                avg_sys=int(round(np.mean(tempo['sys'])/86400))
                self.sys[s_cl+shift]=avg_sys
                self.usr[s_cl+shift]=avg_usr
                self.tot[s_cl+shift]=avg
                print('     ',len(tempo),' avg: ',avg,' client ',avg_usr,' system ',avg_sys)
                #mean+=avg
            	#print('     ',len(tempo),end='')
            shift+=self.n_sub_family
            #print(' avg_tot ',round(mean/self.n_sub_family))
            print('')
        return self
    

    def predict_proba(self, X):
        
        # encode events as frequencies
        data_freqs = self.freq_encoder.transform(X)
        
        # calculate closest clusters for each trace 
        cluster_assignments = self.clustering.predict(data_freqs)
        
        # predict outcomes for each cluster
        cols = [self.case_id_col]+list(['A_Pending','A_Denied','A_Cancelled'])
        preds = pd.DataFrame(columns=cols)
        self.actual = pd.DataFrame(columns=cols)
        shift=0
        self.avg_time_test=np.zeros(self.n_clusters)
        for cl in range(self.n_clusters):
            #print('Distribution of cluster ',cl,':',end='')

            # select cases belonging to given cluster
            cases = data_freqs[cluster_assignments == cl].index
            if len(cases):
                tmp = X[X[self.case_id_col].isin(cases)]
                
                # encode data attributes

                tmp = self.data_encoder.transform(tmp)
                sub_fammily_assignments=self.sub_clustering.predict(tmp[['usr']])
                mean=0
                for s_cl in range(self.n_sub_family):
                    s_cases=tmp[sub_fammily_assignments==s_cl]['Case ID']
                    if len(s_cases)>0:
                        print('cluster ',s_cl+shift,':',end='')

                        tempo=tmp[tmp[self.case_id_col].isin(s_cases)]
                        new_preds = pd.DataFrame(self.clss[s_cl+shift].predict_proba(tempo.drop([self.case_id_col, self.label_col,'sys','usr'], axis=1)))
                        new_preds.columns = self.clss[s_cl+shift].classes_
                        new_preds[self.case_id_col] = tempo.droplevel(1).index
                        preds = pd.concat([preds, new_preds], axis=0, ignore_index=True,sort=False)
                        actuals = pd.get_dummies(tempo[self.label_col])
                        actuals[self.case_id_col] = tempo[self.case_id_col]
                        self.actual = pd.concat([self.actual, actuals], axis=0, ignore_index=True,sort=False)
                        avg=int(round(self.average_in_cluster(tempo)/86400))
                        #mean+=avg
                        avg_usr=int(round(np.mean(tempo['usr'])/86400))
                        avg_sys=int(round(np.mean(tempo['sys'])/86400))
                        if len(s_cases)==1:
                            sys=self.sys[s_cl+shift]-avg_sys
                            usr=self.usr[s_cl+shift]-avg_usr
                            tot=self.tot[s_cl+shift]-avg
                            if sys>=0:
                                print('')
                                print('Remaining time system: ',sys, 'day(s)')
                            else:
                                print('Delay time system: ',-sys, 'day(s)')
                            if usr>=0:
                                print('Remaining time client: ',usr, 'day(s)')
                            else:
                                print('Delay time client: ',-usr, 'day(s)') 
                            if tot>=0:
                                print('Remaining time (total): ',tot, 'day(s)')
                            else:
                                print('Delay time (total): ',-tot, 'day(s)') 


                        print('     ',len(tempo),' avg: ',avg,' client ',avg_usr,' system ',avg_sys)
                        
                        #print('     ',len(tempo),end='')
                    #else:
                    	#print('cluster ',s_cl+shift,':',0,' avg: ',0)
                shift+=self.n_sub_family
                #print(' avg_tot ',round(mean/self.n_sub_family))
                print('')
            else:
                shift+=self.n_sub_family
        preds.fillna(0, inplace=True)
        self.actual.fillna(0, inplace=True)
        #self.actual = self.actual[self.pos_label]
        
        #return preds[self.pos_label]
        return preds 
