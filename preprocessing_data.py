import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from scipy import sparse
import pickle

#def preprocessing_data(features_type_path,train_x_path,train_y_path,test_x_path):
ftype = pd.read_csv('features_type.csv', sep=',')
train_x = pd.read_csv('train_x.csv', sep=',' )
train_y = pd.read_csv('train_y.csv', sep=',')
test_x = pd.read_csv('test_x.csv', sep=',' )

train_x = train_x.drop('uid', axis=1)
train_y = train_y.drop('uid', axis=1)
test_x = test_x.drop('uid', axis=1)

col_numeric = ftype[ftype.type == 'numeric'].feature
col_category = ftype[ftype.type == 'category'].feature

train_n = train_x[col_numeric]
train_c = train_x[col_category]
test_n = test_x[col_numeric]
test_c = test_x[col_category]

train_n = train_n.replace(to_replace='-1', value=np.nan, regex=True)
train_n = train_n.replace(to_replace='-2', value=np.nan, regex=True)
train_c = train_c.replace(to_replace='-1', value=np.nan, regex=True)
train_c = train_c.replace(to_replace='-2', value=np.nan, regex=True)
test_n = test_n.replace(to_replace='-1', value=np.nan, regex=True)
test_n = test_n.replace(to_replace='-2', value=np.nan, regex=True)
test_c = test_c.replace(to_replace='-1', value=np.nan, regex=True)
test_c = test_c.replace(to_replace='-2', value=np.nan, regex=True)

count_n = train_n.count(axis=0)
count_c = train_c.count(axis=0)
bad_nfeature = count_n[count_n.values<5000].index
bad_cfeature = count_c[count_c.values<5000].index
mean_nfeature = count_n[(count_n.values>5000)&(count_n.values<10000)].index
freq_nfeature = count_c[(count_c.values>5000)&(count_c.values<10000)].index
pred_nfeature = count_n[(count_n.values>10000)&(count_n.values<15000)].index
pred_cfeature = count_c[(count_c.values>10000)&( count_c.values<15000)].index

train_n = train_n.drop(bad_nfeature,axis=1)
train_c = train_c.drop(bad_cfeature,axis=1)
test_n = test_n.drop(bad_nfeature,axis=1)
test_c = test_c.drop(bad_cfeature,axis=1)

#dummies
#x1107	x1108	x1109	x1110	x1111	x1112	x1113	x1114	x1115	
#x1116	x1117	x1118	x1119	x1120	x1121	x1122	x1123	x1124	
#x1125	x1126	x1127	x1128	x1129	x1130	x1131	x1132	x1133	
#x1134	x1135	x1136	x1137	x1138
dummies_label = ['x1107','x1108','x1109','x1110','x1111','x1112','x1113','x1114','x1115',
                 'x1116','x1117','x1118','x1119','x1120','x1121','x1122','x1123','x1124',
                 'x1125','x1126','x1127','x1128','x1129','x1130','x1131','x1132','x1133',
                 'x1134','x1135','x1136','x1137','x1138']
for dlabel in dummies_label:
	tmp = pd.get_dummies(train_c[dlabel], prefix=dlabel)
	train_c = pd.concat([train_c, tmp], axis=1)
	train_c = train_c.drop(dlabel,axis=1)
	tmp = pd.get_dummies(test_c[dlabel], prefix=dlabel)
	test_c = pd.concat([test_c, tmp], axis=1)
	test_c = test_c.drop(dlabel,axis=1)

#Transform train_c to sparse 
vec = DictVectorizer()
trainc_sparse = vec.fit_transform(train_c.T.to_dict().values())
testc_sparse = vec.fit_transform(test_c.T.to_dict().values())

#Scaling features
#min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-3,3))
#trainn_scale = min_max_scaler.fit_transform(train_n)
trian_n = train_n.fillna(-9999)
x=sparse.csr_matrix(sparse.hstack([sparse.coo_matrix(trian_n),sparse.coo_matrix(trainc_sparse)]))
test_n = test_n.fillna(-9999)
tx=sparse.csr_matrix(sparse.hstack([sparse.coo_matrix(test_n),sparse.coo_matrix(testc_sparse)]))

y = train_y.as_matrix(columns=None)

pickle.dump(x,open('X.p','wb'))
pickle.dump(tx,open('TX.p','wb'))
pickle.dump(y,open('y.p','wb'))












#train_c.to_csv('train_c_dummies.csv', delimiter=',', index_col=False)
#train_n.to_csv('train_n.csv', delimiter=',', index_col=False)
#train_c.to_csv('train_c.csv', delimiter=',', index_col=False)
