```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

music_data = pd.read_csv('music.csv')
X = music_data.drop(columns=['genre'])
y = music_data['genre']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
score = accuracy_score(y_test, predictions)
score
```




    1.0




```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


# music_data = pd.read_csv('music.csv')
# X = music_data.drop(columns=['genre'])
# y = music_data['genre']

# model = DecisionTreeClassifier()
# model.fit(X, y)

joblib.load('music-recommender.joblib')
predictions = model.predict([[21, 1]])
predictions
```

    C:\Users\q1\AppData\Local\Programs\Python\Python311\Lib\site-packages\sklearn\base.py:464: UserWarning: X does not have valid feature names, but DecisionTreeClassifier was fitted with feature names
      warnings.warn(
    




    array(['HipHop'], dtype=object)




```python
from sklearn import tree
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


music_data = pd.read_csv('music.csv')
X = music_data.drop(columns=['genre'])
y = music_data['genre']


model = DecisionTreeClassifier()
model.fit(X, y)

tree.export_graphviz(model, out_file= 'music-recommender.dot',
                    feature_names= ['age', 'gender'],
                    class_names= sorted(y.unique()),
                    label= 'all',
                    rounded= True,
                    filled= True)
```


```python

```


```python

```
