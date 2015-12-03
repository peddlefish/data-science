#!/usr/bin/python
# -*- coding: utf-8 -*-

from sklearn.svm import SVC
import numpy as np
import pandas as pd
def main():
	## read training dataset
	traindataset = pd.read_csv('/usr3/graduate/xysun/walmart/traindata.csv')
	target = traindataset['TripType']
	traindata_feature = traindataset.drop(['TripType','VisitNumber'],1)
	svc = SVC()
	svc.fit(traindata_feature, target)

	
	##test models
	testdataset = pd.read_csv('/usr3/graduate/xysun/walmart/testdata.csv')
	indexn = testdataset['VisitNumber']
	traindata = traindataset.drop(['VisitNumber'],1)
	result = svc.predict(testdata)
	svc_csv = pd.DataFrame(result[0:,0:], index=indexn)
	svc_csv.index.name = "VisitNumber"
	svc_csv.columns = ["TripType_3","TripType_4","TripType_5","TripType_6","TripType_7","TripType_8","TripType_9","TripType_12","TripType_14","TripType_15","TripType_18","TripType_19","TripType_20","TripType_21","TripType_22","TripType_23","TripType_24","TripType_25","TripType_26","TripType_27","TripType_28","TripType_29","TripType_30","TripType_31","TripType_32","TripType_33","TripType_34","TripType_35","TripType_36","TripType_37","TripType_38","TripType_39","TripType_40","TripType_41","TripType_42","TripType_43","TripType_44","TripType_999"]
	svc_csv.to_csv('/usr3/graduate/xysunn/walmart/svc_csv.csv',header=True, index=True,delimiter=',')

if __name__=="__main__":
	main()


