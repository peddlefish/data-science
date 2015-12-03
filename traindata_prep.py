import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import time
def main():
	start = time.clock()
	COMPONENT_NUM = 100
	## read training dataset
	traindataset = pd.read_csv('/usr3/graduate/xysun/walmart-classification/train.csv')
	traindataset.dropna(how='any',inplace=True)

	## data preparation 
	# convert categorical variable into dummy variables
	df_weekday = pd.get_dummies(traindataset['Weekday'])
	traindataset1 = traindataset.join(df_weekday)
	traindataset1.drop('Weekday', axis=1, inplace=True)

	# merge lines of one VisitNumber in one line
	traindata_by_visitnumber_first = traindataset1.groupby('VisitNumber').first()
	# create new field: if the customer return a product
	traindata_by_visitnumber_first['boolean_of_return'] = traindataset1.groupby('VisitNumber').min()['ScanCount'].apply(lambda x: x==-1)
	# create new fields: the amount of the items types each customer bought or return: positive 
	traindataset1['ScanCount_sign']=np.sign(traindataset1.ScanCount)
	amount_of_itemstp = traindataset1.groupby('VisitNumber').ScanCount_sign.value_counts().unstack()
	amount_of_itemstp.fillna(0,inplace=True)
	amount_of_itemstp.columns = ['amount_of_returntp', 'amount_of_buytp']
	traindata_by_visitnumber1 = traindata_by_visitnumber_first.join(amount_of_itemstp)
	# create new fields: Department Description Matrix for each visitnumber:
	Fineline_Matrix = pd.pivot_table(traindataset1, values='ScanCount', index=['VisitNumber'],columns='FinelineNumber',aggfunc=np.sum)
	Fineline_Matrix.fillna(0,inplace=True)
	traindata_by_visitnumber = traindata_by_visitnumber1.join(Fineline_Matrix)
	print traindata_by_visitnumber.shape
	print traindata_by_visitnumber.dtypes
	traindata = traindata_by_visitnumber.drop(['Upc','ScanCount','DepartmentDescription','FinelineNumber'],1)
	pca = PCA(n_components=COMPONENT_NUM, whiten=True)
	pca.fit(traindata)
	traindata = pca.transform(traindata)
	data_prep_time = time.clock() - start
	traindata.to_csv('/usr3/graduate/xysun/walmart-classification/traindata.csv',header=True, index=True,delimiter=',')
	print data_prep_time

	##test models
	testdataset = pd.read_csv('/usr3/graduate/xysun/walmart-classification/test.csv')
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
	amount_of_itemstp_t.columns = ['amount_of_returntp_t', 'amount_of_buytp_t']
	testdata_by_visitnumber1 = testdata_by_visitnumber_first.join(amount_of_itemstp_t)
	# create new fields: Department Description Matrix for each visitnumber:
	Fineline_Matrix_t = pd.pivot_table(testdataset1, values='ScanCount', index=['VisitNumber'],columns='FinelineNumber',aggfunc=np.sum)
	Fineline_Matrix_t.fillna(0,inplace=True)
	testdata_by_visitnumber = testdata_by_visitnumber1.join(Fineline_Matrix_t)
	testdata = testdata_by_visitnumber.drop(['Upc','ScanCount','DepartmentDescription','FinelineNumber'],1)
	testdata.fillna(0,inplace=True)
	testdata = pca.transform(testdata)
	testdata.to_csv('/usr3/graduate/xysun/walmart-classification/testdata.csv',header=True,index=True,delimiter=',')
if __name__=="__main__":
	main()
