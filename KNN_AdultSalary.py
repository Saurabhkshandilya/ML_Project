import pandas as pd
obj=pd.read_excel(open('adult_salary_dataset.xlsx','rb'))

x=obj.iloc[:,:-1]
y=obj.iloc[:,-1]

import numpy as np
x=np.array(x)
y=np.array(y)


from sklearn.impute import SimpleImputer
#im = SimpleImputer(missing_values=np.nan, strategy='mean')
im = SimpleImputer(missing_values='?',strategy='most_frequent')
x=im.fit_transform(x)


from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
transformer = ColumnTransformer(
    transformers=[
        ("OneHot",        # Just a name
         OneHotEncoder(), # The transformer class
         [1,3,5,6,7,8,9,11]              # The column(s) to be applied on.
         )
    ],
    remainder='passthrough' # donot apply anything to the remaining columns
)

x = transformer.fit_transform(x)


labencode=LabelEncoder()
y=labencode.fit_transform(y)


from sklearn.preprocessing import MaxAbsScaler
sc=MaxAbsScaler()
x=x.toarray()
x=sc.fit_transform(x)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)


#Few ML classification algos
from sklearn.neighbors import KNeighborsClassifier as knn
knnobject=knn()
knnobject.fit(xtrain,ytrain)
ypred=knnobject.predict(xtest)
ypred

from sklearn.metrics import accuracy_score
acc=accuracy_score(ypred,ytest)
acc


















