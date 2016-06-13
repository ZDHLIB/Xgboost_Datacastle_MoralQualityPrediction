import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 

#class figure_plot:

def auc_curvePlot(auc, title='AUC curves'):

	test_auc_mean = auc['test-auc-mean']
	test_auc_std = auc['test-auc-std']
	train_auc_mean = auc['train-auc-mean']
	train_auc_std = auc['train-auc-std']

	x = np.linspace(1,len(auc),len(auc))
	plt.figure()
	plt.title('AUC curves')
	plt.xlabel(u"train number")
	plt.ylabel(u"auc")
	plt.grid()

	plt.fill_between(x, train_auc_mean - train_auc_std, train_auc_mean + train_auc_std, alpha=0.1, color="b")
	plt.fill_between(x, test_auc_mean - test_auc_std, test_auc_mean + test_auc_std, alpha=0.1, color="r")
	plt.plot(x, train_auc_mean, 'o-', color="b", label=u'train_auc_mean')
	plt.plot(x, test_auc_mean, 'o-', color="r", label=u'test_auc_mean')

#1
auc = pd.read_csv('cv_0.01_0.1_3_8_0.7_300_1100-15000_1_6000.csv')
test_auc_mean = auc['test-auc-mean']
test_auc_std = auc['test-auc-std']
train_auc_mean = auc['train-auc-mean']
train_auc_std = auc['train-auc-std']
x = np.linspace(1,len(auc),len(auc))
plt.fill_between(x, train_auc_mean - train_auc_std, train_auc_mean + train_auc_std, alpha=0.1, color="b")
plt.fill_between(x, test_auc_mean - test_auc_std, test_auc_mean + test_auc_std, alpha=0.1, color="b")
plt.plot(x, train_auc_mean, 'o-', color="b", label=u'train_auc_mean')
plt.plot(x, test_auc_mean, 'o-', color="b", label=u'test_auc_mean')

#2
'''
auc = pd.read_csv('cv_0.01_0.1_3_8_0.7_400_1100-15000_1_6000.csv')
test_auc_mean = auc['test-auc-mean']
test_auc_std = auc['test-auc-std']
train_auc_mean = auc['train-auc-mean']
train_auc_std = auc['train-auc-std']
x = np.linspace(1,len(auc),len(auc))
plt.fill_between(x, train_auc_mean - train_auc_std, train_auc_mean + train_auc_std, alpha=0.1, color="r")
plt.fill_between(x, test_auc_mean - test_auc_std, test_auc_mean + test_auc_std, alpha=0.1, color="r")
plt.plot(x, train_auc_mean, 'o-', color="r", label=u'train_auc_mean')
plt.plot(x, test_auc_mean, 'o-', color="r", label=u'test_auc_mean')


#3
auc = pd.read_csv('cv_0.01_0.1_4_8_0.7_500_1100-15000_1_6000.csv')
test_auc_mean = auc['test-auc-mean']
test_auc_std = auc['test-auc-std']
train_auc_mean = auc['train-auc-mean']
train_auc_std = auc['train-auc-std']
x = np.linspace(1,len(auc),len(auc))
plt.fill_between(x, train_auc_mean - train_auc_std, train_auc_mean + train_auc_std, alpha=0.1, color="r")
plt.fill_between(x, test_auc_mean - test_auc_std, test_auc_mean + test_auc_std, alpha=0.1, color="r")
plt.plot(x, train_auc_mean, 'o-', color="r", label=u'train_auc_mean')
plt.plot(x, test_auc_mean, 'o-', color="r", label=u'test_auc_mean')


#4
auc = pd.read_csv('cv_0.01_0.1_3_8_0.7_200_1100-15000_1_6000.csv')
test_auc_mean = auc['test-auc-mean']
test_auc_std = auc['test-auc-std']
train_auc_mean = auc['train-auc-mean']
train_auc_std = auc['train-auc-std']
x = np.linspace(1,len(auc),len(auc))
plt.fill_between(x, train_auc_mean - train_auc_std, train_auc_mean + train_auc_std, alpha=0.1, color="r")
plt.fill_between(x, test_auc_mean - test_auc_std, test_auc_mean + test_auc_std, alpha=0.1, color="r")
plt.plot(x, train_auc_mean, 'o-', color="r", label=u'train_auc_mean')
plt.plot(x, test_auc_mean, 'o-', color="r", label=u'test_auc_mean')

#5
auc = pd.read_csv('cv_0.01_0.1_3_8_0.7_100_1100-15000_1_6000.csv')
test_auc_mean = auc['test-auc-mean']
test_auc_std = auc['test-auc-std']
train_auc_mean = auc['train-auc-mean']
train_auc_std = auc['train-auc-std']
x = np.linspace(1,len(auc),len(auc))
plt.fill_between(x, train_auc_mean - train_auc_std, train_auc_mean + train_auc_std, alpha=0.1, color="r")
plt.fill_between(x, test_auc_mean - test_auc_std, test_auc_mean + test_auc_std, alpha=0.1, color="r")
plt.plot(x, train_auc_mean, 'o-', color="r", label=u'train_auc_mean')
plt.plot(x, test_auc_mean, 'o-', color="r", label=u'test_auc_mean')

#6
auc = pd.read_csv('cv_0.01_0.1_3_8_0.7_1_1100-15000_1_6000.csv')
test_auc_mean = auc['test-auc-mean']
test_auc_std = auc['test-auc-std']
train_auc_mean = auc['train-auc-mean']
train_auc_std = auc['train-auc-std']
x = np.linspace(1,len(auc),len(auc))
plt.fill_between(x, train_auc_mean - train_auc_std, train_auc_mean + train_auc_std, alpha=0.1, color="r")
plt.fill_between(x, test_auc_mean - test_auc_std, test_auc_mean + test_auc_std, alpha=0.1, color="r")
plt.plot(x, train_auc_mean, 'o-', color="r", label=u'train_auc_mean')
plt.plot(x, test_auc_mean, 'o-', color="r", label=u'test_auc_mean')


#7
auc = pd.read_csv('cv_0.01_0.1_3_10_0.7_300_1100-15000_8_6000.csv')
test_auc_mean = auc['test-auc-mean']
test_auc_std = auc['test-auc-std']
train_auc_mean = auc['train-auc-mean']
train_auc_std = auc['train-auc-std']
x = np.linspace(1,len(auc),len(auc))
plt.fill_between(x, train_auc_mean - train_auc_std, train_auc_mean + train_auc_std, alpha=0.1, color="r")
plt.fill_between(x, test_auc_mean - test_auc_std, test_auc_mean + test_auc_std, alpha=0.1, color="r")
plt.plot(x, train_auc_mean, 'o-', color="r", label=u'train_auc_mean')
plt.plot(x, test_auc_mean, 'o-', color="r", label=u'test_auc_mean')
'''

#8
auc = pd.read_csv('cv_0.01_0.1_3_13_0.7_300_1100-15000_8_6000.csv')
test_auc_mean = auc['test-auc-mean']
test_auc_std = auc['test-auc-std']
train_auc_mean = auc['train-auc-mean']
train_auc_std = auc['train-auc-std']
x = np.linspace(1,len(auc),len(auc))
plt.fill_between(x, train_auc_mean - train_auc_std, train_auc_mean + train_auc_std, alpha=0.1, color="r")
plt.fill_between(x, test_auc_mean - test_auc_std, test_auc_mean + test_auc_std, alpha=0.1, color="r")
plt.plot(x, train_auc_mean, 'o-', color="r", label=u'train_auc_mean')
plt.plot(x, test_auc_mean, 'o-', color="r", label=u'test_auc_mean')

'''Choose 1,7 and 8'''


plt.show()