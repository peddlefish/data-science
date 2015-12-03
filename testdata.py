import numpy as np
import pandas as pd
import time
def main():
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
	testdata.to_csv('/usr3/graduate/xysun/walmart-classification/testdata.csv',header=True,index=True,delimiter=',')

if __name__=="__main__":
	main()