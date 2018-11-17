#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.learning_curve import validation_curve

def PolynomialRegression(degree=2, **kwargs):
	return make_pipeline(PolynomialFeatures(degree), 
	                     LinearRegression(**kwargs))


	
def make_data(N, err=1.0, rseed=1):
	rng = np.random.RandomState(rseed)#随机抽样数据
	X = rng.rand(N, 1) ** 2
	y = 10 - 1. / (X.ravel() + 0.1)
	if err > 0:
		y += err * rng.randn(N)
	return X, y

if __name__ == "__main__":
	X, y = make_data(40)
	sns.set()
	X_test = np.linspace(-0.1, 1.1, 500)[:, None]  #[:,None]将维度(ndim,)转为(ndim,1)
	plt.scatter(X.ravel(), y, color='black')
	axis = plt.axis()
	for degree in [1, 3, 5]:
		y_test = PolynomialRegression(degree).fit(X, y).predict(X_test)
		plt.plot(X_test.ravel(), y_test, label='degree({0})'.format(degree))
	plt.xlim(-0.1, 1.0)
	plt.ylim(-2, 12)
	plt.legend(loc='best')
	

#可视化验证曲线
	fig = plt.figure()
	
	degree = np.arange(0, 21)
	train_score, val_score = validation_curve(PolynomialRegression(), X, y,
											'polynomialfeatures__degree',
											degree, cv=7)
	plt.plot(degree, np.median(train_score, 1), color='blue', label='trainning score')
	plt.plot(degree, np.median(val_score, 1), color='red', label='validation score')
	plt.legend(loc='best')
	plt.ylim(0, 1)
	plt.xlabel('degree')
	plt.ylabel('score')
#3次多项式的验证模型是最好的
	fig = plt.subplots()
	plt.scatter(X.ravel(), y)
	lim = plt.axis()
	y_test = PolynomialRegression(3).fit(X, y).predict(X_test)
	plt.plot(X_test.ravel(), y_test)
	plt.axis(lim)
	plt.show()


	