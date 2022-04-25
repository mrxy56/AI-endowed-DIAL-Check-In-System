from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C= 1000, penalty= ‘l1’)
clf.fit(x_train,y_train)
pred = clf.predict(x_test)
print(“Non Zero weights:”,np.count_nonzero(clf.coef_))
