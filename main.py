from xgb_class import xgb_class
import numpy as np
import pandas as pd
import pickle
from email.mime.text import MIMEText
import smtplib
'''
def send_mail(to_list,sub,content):
    mail_host="smtp.163.com"
    mail_user="zhongdunhao"
    mail_pass="zdh89786"
    mail_postfix="163.com"
    me=mail_user+"<"+mail_user+"@"+mail_postfix+">"
    msg = MIMEText(content)
    msg['Subject'] = sub
    msg['From'] = me
    msg['To'] = to_list
    try:
        s = smtplib.SMTP()
        s.connect(mail_host)
        s.login(mail_user,mail_pass)
        s.sendmail(me, to_list, msg.as_string())
        s.close()
        print '1'
        return True
    except Exception, e:
        print '2'
        print str(e)
        return False

y = pickle.load(open('y.p','rb'))
x = pickle.load(open('X.p','rb'))
tx = pickle.load(open('TX.p','rb'))


#cv_0.01_0.1_4_8_0.7_500_1100-15000_1_5000
clf = xgb_class(silent=0,eta=0.01,gamma=0.1,min_chile_weight=3,max_depth=8,subsample=0.7,lambda_=500,scale_pos_weight=1100.0/15000,
	              colsample_bytree=1,num_boost_round=6000,exist_prediction=0,exist_num_boost_round=20,threads=10) 
cv_res = clf.train_crossV(x,y, nfold=5, early_stopping_rounds=300, metrics=['auc'])
cv_res.to_csv('cv_0.01_0.1_4_8_0.7_500_1100-15000_1_6000.csv', dep=',')


clf = xgb_class(silent=0,eta=0.01,gamma=0.1,min_chile_weight=3,max_depth=8,subsample=0.7,lambda_=400,scale_pos_weight=1100.0/15000,
	              colsample_bytree=1,num_boost_round=6000,exist_prediction=0,exist_num_boost_round=20,threads=10) 
cv_res = clf.train_crossV(x,y, nfold=5, early_stopping_rounds=300, metrics=['auc'])
cv_res.to_csv('cv_0.01_0.1_3_8_0.7_400_1100-15000_1_6000.csv', dep=',')


clf = xgb_class(silent=0,eta=0.01,gamma=0.1,min_chile_weight=3,max_depth=8,subsample=0.7,lambda_=300,scale_pos_weight=1100.0/15000,
	              colsample_bytree=1,num_boost_round=6000,exist_prediction=0,exist_num_boost_round=20,threads=10) 
cv_res = clf.train_crossV(x,y, nfold=5, early_stopping_rounds=300, metrics=['auc'])
cv_res.to_csv('cv_0.01_0.1_3_8_0.7_300_1100-15000_1_6000.csv', dep=',')


clf = xgb_class(silent=0,eta=0.01,gamma=0.1,min_chile_weight=3,max_depth=8,subsample=0.7,lambda_=200,scale_pos_weight=1100.0/15000,
	              colsample_bytree=1,num_boost_round=6000,exist_prediction=0,exist_num_boost_round=20,threads=10) 
cv_res = clf.train_crossV(x,y, nfold=5, early_stopping_rounds=300, metrics=['auc'])
cv_res.to_csv('cv_0.01_0.1_3_8_0.7_200_1100-15000_1_6000.csv', dep=',')


clf = xgb_class(silent=0,eta=0.01,gamma=0.1,min_chile_weight=3,max_depth=8,subsample=0.7,lambda_=100,scale_pos_weight=1100.0/15000,
	              colsample_bytree=1,num_boost_round=6000,exist_prediction=0,exist_num_boost_round=20,threads=10) 
cv_res = clf.train_crossV(x,y, nfold=5, early_stopping_rounds=300, metrics=['auc'])
cv_res.to_csv('cv_0.01_0.1_3_8_0.7_100_1100-15000_1_6000.csv', dep=',')


clf = xgb_class(silent=0,eta=0.01,gamma=0.1,min_chile_weight=3,max_depth=8,subsample=0.7,lambda_=1,scale_pos_weight=1100.0/15000,
	              colsample_bytree=1,num_boost_round=6000,exist_prediction=0,exist_num_boost_round=20,threads=10) 
cv_res = clf.train_crossV(x,y, nfold=5, early_stopping_rounds=300, metrics=['auc'])
cv_res.to_csv('cv_0.01_0.1_3_8_0.7_1_1100-15000_1_6000.csv', dep=',')


clf = xgb_class(silent=0,eta=0.01,gamma=0.1,min_chile_weight=3,max_depth=10,subsample=0.7,lambda_=300,scale_pos_weight=1100.0/15000,
	              colsample_bytree=0.8,num_boost_round=6000,exist_prediction=0,exist_num_boost_round=20,threads=10) 
cv_res = clf.train_crossV(x,y, nfold=5, early_stopping_rounds=200, metrics=['auc'])
cv_res.to_csv('cv_0.01_0.1_3_10_0.7_300_1100-15000_8_6000.csv', dep=',')


clf = xgb_class(silent=0,eta=0.01,gamma=0.1,min_chile_weight=3,max_depth=13,subsample=0.7,lambda_=300,scale_pos_weight=1100.0/15000,
	              colsample_bytree=0.8,num_boost_round=6000,exist_prediction=0,exist_num_boost_round=20,threads=10) 
cv_res = clf.train_crossV(x,y, nfold=5, early_stopping_rounds=200, metrics=['auc'])
cv_res.to_csv('cv_0.01_0.1_3_13_0.7_300_1100-15000_8_6000.csv', dep=',')


clf = xgb_class(silent=0,eta=0.01,gamma=0.1,min_chile_weight=3,max_depth=18,subsample=0.7,lambda_=300,scale_pos_weight=1100.0/15000,
	              colsample_bytree=0.8,num_boost_round=6000,exist_prediction=0,exist_num_boost_round=20,threads=10) 
cv_res = clf.train_crossV(x,y, nfold=5, early_stopping_rounds=200, metrics=['auc'])
cv_res.to_csv('cv_0.01_0.1_3_18_0.7_300_1100-15000_8_6000.csv', dep=',')
'''

'''Choose 1,7 and 8'''
'''
#1
clf = xgb_class(silent=0,eta=0.01,gamma=0.1,min_chile_weight=3,max_depth=8,subsample=0.7,lambda_=300,scale_pos_weight=1100.0/15000,
	              colsample_bytree=1,num_boost_round=15000,exist_prediction=True,exist_num_boost_round=10000,threads=20) 
ypred1 = clf.train_predict(train_x=x,train_y=y,test_x=tx)
np.savetxt('1.csv',ypred1, delimiter=',')

#7
clf = xgb_class(silent=0,eta=0.01,gamma=0.1,min_chile_weight=3,max_depth=10,subsample=0.7,lambda_=300,scale_pos_weight=1100.0/15000,
	              colsample_bytree=0.8,num_boost_round=15000,exist_prediction=True,exist_num_boost_round=10000,threads=20) 
ypred7 = clf.train_predict(train_x=x,train_y=y,test_x=tx)
np.savetxt('7.csv',ypred7, delimiter=',')

#8
clf = xgb_class(silent=0,eta=0.01,gamma=0.1,min_chile_weight=3,max_depth=13,subsample=0.7,lambda_=300,scale_pos_weight=1100.0/15000,
	              colsample_bytree=0.8,num_boost_round=15000,exist_prediction=True,exist_num_boost_round=10000,threads=20) 
ypred8 = clf.train_predict(train_x=x,train_y=y,test_x=tx)
np.savetxt('8.csv',ypred8, delimiter=',')

res = (ypred1 + ypred7 + ypred8)/3.0
np.savetxt('ensemble.csv',res, delimiter=',')

send_mail("zhongdunhao@163.com","Datacastle has finished","Let's have a look.")
'''

eve = np.loadtxt(file('ensemble.csv'))
best048 = pd.read_csv('best_72048.csv', sep=',')
best113 = pd.read_csv('best_72113.csv', sep=',')
best048 = best048['score'].values
best113 = best113['score'].values

res = (eve * 0.3) + ((best048 + best113)*0.5 * 0.7)

np.savetxt('eve3_best7.csv', res, delimiter=',')