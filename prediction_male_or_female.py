from sklearn import tree

# [hieght,weight,shoe size]

x = [[150,80,44],[166,95,48],[188,88,45],[175,75,50]]

y = ['male','female','female','male']

clf = tree.DecisionTreeClassifier()

clf = clf.fit(x,y)

prediction = clf.predict([[175,75,50]])

print(prediction)
