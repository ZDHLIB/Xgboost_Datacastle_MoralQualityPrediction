import xgboost as xgb
import numpy as np

class xgb_class:

	def __init__(self,silent,eta,gamma,min_chile_weight,max_depth,subsample,lambda_,scale_pos_weight,
		         colsample_bytree,num_boost_round,exist_prediction=0,exist_num_boost_round=20,threads=8):
		self.silent=silent
		self.eta=eta
		self.gamma=gamma
		self.min_chile_weight=min_chile_weight
		self.max_depth=max_depth
		self.subsample=subsample
		self.lambda_=lambda_
		self.scale_pos_weight=scale_pos_weight
		self.colsample_bytree=colsample_bytree
		self.num_boost_round=num_boost_round
		self.exist_prediction=exist_prediction
		self.exist_num_boost_round=exist_num_boost_round
		self.threads=threads

	def train_predict(self,train_x,train_y,test_x):
		xgmat_train = xgb.DMatrix(train_x, label=train_y, missing=-9999)
		test_size = test_x.shape[0]
		params = {
			'booster':'gbtree',
			'objective':'binary:logistic',
			'silent':self.silent,
			'eta':self.eta,
			'gamma':self.gamma,
			'max_depth':self.max_depth,
			'min_chile_weitght':self.min_chile_weight,
			'subsample':self.subsample,
			'lambda':self.lambda_,
			'scale_pos_weight':self.scale_pos_weight,
			"colsample_bytree": self.colsample_bytree,
			'eval_metirc':'auc',
			'seed':2014,
			'nthread':self.threads
		}

		watchlist = [ (xgmat_train,'train') ]
		num_round = self.num_boost_round

		bst = xgb.train( params, xgmat_train, num_round, watchlist )
		xgmat_test = xgb.DMatrix(test_x,missing=-9999)

		if self.exist_prediction:
			tmp_train = bst.predict(xgmat_train, output_margin=True)
			tmp_test = bst.predict(xgmat_test, output_margin=True)
			xgmat_train.set_base_margin(tmp_train)
			xgmat_test.set_base_margin(tmp_test)
			bst = xgb.train(params, xgmat_train, self.exist_num_boost_round, watchlist )

		ypred = bst.predict(xgmat_test)
		return ypred

	def train_crossV(self, train_x, train_y, nfold=3, early_stopping_rounds=300, metrics=['auc']):
		xgmat_train = xgb.DMatrix(train_x, label=train_y, missing=-9999)

		params = {
			'booster':'gbtree',
			'objective':'binary:logistic',
			'silent':self.silent,
			'eta':self.eta,
			'gamma':self.gamma,
			'max_depth':self.max_depth,
			'min_chile_weitght':self.min_chile_weight,
			'subsample':self.subsample,
			'lambda':self.lambda_,
			'scale_pos_weight':self.scale_pos_weight,
			"colsample_bytree": self.colsample_bytree,
			'eval_metirc':'auc',
			'seed':2014,
			'nthread':self.threads
		}

		watchlist = [ (xgmat_train,'train') ]
		num_round = self.num_boost_round

		cv_result = xgb.cv(params, xgmat_train, num_boost_round=num_round, early_stopping_rounds=early_stopping_rounds, nfold=nfold, seed=1024, show_progress=True, metrics=metrics)

		return cv_result


    

