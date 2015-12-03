import sys
sys.path.append('/Users/Fanghui/Desktop/xgboost-master/wrapper/')
import xgboost as xgb

import numpy as np
import pandas as pd
import time
def main():
	# print dir(xgbs.xgb)
	dtrain = xgb.DMatrix('/Users/Fanghui/Desktop/walmart-classification/traindata.csv')
	# dtest = xgboost.DMatrix('/Users/Fanghui/Desktop/walmart-classification/testdata.csv')
	print type(dtrain)
	
if __name__=="__main__":
	main()
