#! /usr/bin/python
# -*- coding: utf-8 -*-
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn import tree 
import numpy as np
import pandas as pd
import time
def main():
	start = time.clock()
	## read training dataset
	traindataset = pd.read_csv('/usr3/graduate/xysun/walmart/train.csv')
 	# delete rows with missing data shape=(647054,1)
	traindataset.dropna(how='any',inplace=True)


	## data preparation 
	# convert categorical variable into dummy variables
	df_weekday = pd.get_dummies(traindataset['Weekday'])
	traindataset1 = traindataset.join(df_weekday)
	traindataset1.drop('Weekday', axis=1, inplace=True)

	# merge lines of one VisitNumber in one line
	traindata_by_visitnumber_first = traindataset1.groupby('VisitNumber').first()
	target = traindata_by_visitnumber_first['TripType']
	# create new field: if the customer return a product
	traindata_by_visitnumber_first['boolean_of_return'] = traindataset1.groupby('VisitNumber').min()['ScanCount'].apply(lambda x: x==-1)
	# create new fields: the amount of the items types each customer bought or return: positive 
	traindataset1['ScanCount_sign']=np.sign(traindataset1.ScanCount)
	amount_of_itemstp = traindataset1.groupby('VisitNumber').ScanCount_sign.value_counts().unstack()
	amount_of_itemstp.fillna(0,inplace=True)
	amount_of_itemstp.columns = ['amount_of_buytp', 'amount_of_returntp']
	traindata_by_visitnumber1 = traindata_by_visitnumber_first.join(amount_of_itemstp)
	# create new fields: Department Description Matrix for each visitnumber:
	Fineline_Matrix = pd.pivot_table(traindataset1, values='ScanCount',index=['VisitNumber'],columns='FinelineNumber',aggfunc=np.sum)
	Fineline_Matrix.fillna(0,inplace=True)
	traindata_by_visitnumber = traindata_by_visitnumber1.join(Fineline_Matrix)

	#print traindata_by_visitnumber.dtypes
	##fit models: Random Forest
	data_prep_time = time.clock() - start
	rfc_traindata = traindata_by_visitnumber.drop(['TripType','Upc','ScanCount','DepartmentDescription','FinelineNumber'],1)
	rfc = RandomForestClassifier(n_estimators=1000)
	rfc = rfc.fit(rfc_traindata, target)
	Dtree_fitting_time = time.clock() - data_prep_time-start
	print Dtree_fitting_time

	
	##test models
	testdataset = pd.read_csv('/usr3/graduate/xysun/walmart/test.csv')
	df_weekday_t = pd.get_dummies(testdataset['Weekday'])
	testdataset1 = testdataset.join(df_weekday_t)
	#how to deal with missing data in test datasets
	testdataset1.fillna(0,inplace=True)
	testdataset1.drop('Weekday', axis=1, inplace=True)
	testdata_by_visitnumber_first = testdataset1.groupby('VisitNumber').first()
	# create new field: if the customer return a product
	testdata_by_visitnumber_first['boolean_of_return'] = testdataset1.groupby('VisitNumber').min()['ScanCount'].apply(lambda x: x==-1)
	# create new fields: the amount of the items types each customer bought or return: positive 
	testdataset1['ScanCount_sign']=np.sign(testdataset1.ScanCount)
	amount_of_itemstp_t = testdataset1.groupby('VisitNumber').ScanCount_sign.value_counts().unstack()
	amount_of_itemstp_t.fillna(0,inplace=True)
	amount_of_itemstp_t.columns = ['amount_of_buytp_t', 'amount_of_returntp_t']
	testdata_by_visitnumber1 = testdata_by_visitnumber_first.join(amount_of_itemstp_t)
	# create new fields: Department Description Matrix for each visitnumber:
	Fineline_Matrix_t = pd.pivot_table(testdataset1, values='ScanCount', index=['VisitNumber'],columns='FinelineNumber',aggfunc=np.sum)
	Fineline_Matrix_t.fillna(0,inplace=True)
	testdata_by_visitnumber = testdata_by_visitnumber1.join(Fineline_Matrix_t)
	rfc_testdata = testdata_by_visitnumber.drop(['Upc','ScanCount','DepartmentDescription','FinelineNumber'],1)
	rfc_testdata.fillna(0,inplace=True)

	result = rfc.predict_proba(rfc_testdata)

	indexn = testdata_by_visitnumber_first.index
	#testdata_by_visitnumber_first['VisitNumber']
	result_csv2 = pd.DataFrame(result[0:,0:], index=indexn)
	print result_csv2.shape
	result_csv2.index.name = "VisitNumber"
	result_csv2.columns = ["TripType_3","TripType_4","TripType_5","TripType_6","TripType_7","TripType_8","TripType_9","TripType_12","TripType_14","TripType_15","TripType_18","TripType_19","TripType_20","TripType_21","TripType_22","TripType_23","TripType_24","TripType_25","TripType_26","TripType_27","TripType_28","TripType_29","TripType_30","TripType_31","TripType_32","TripType_33","TripType_34","TripType_35","TripType_36","TripType_37","TripType_38","TripType_39","TripType_40","TripType_41","TripType_42","TripType_43","TripType_44","TripType_999"]
	print result_csv2.dtypes
	result_csv2.to_csv('/usr3/graduate/xysun/walmart/result_rf_fl.csv',header=True, index=True,delimiter=',')

if __name__=="__main__":
	main()

