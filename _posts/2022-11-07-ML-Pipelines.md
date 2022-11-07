---
layout: post
title: ML Pipelines
---


<div id="container" style="position:relative;">
<div style="float:left"><h1> ML Pipelines </h1></div>
<div style="position:relative; float:right"><img style="height:65px" src ="https://drive.google.com/uc?export=view&id=1EnB0x-fdqMp6I5iMoEBBEuxB_s7AmE2k" />
</div>
</div>




So far, we have seen the steps in creating a machine learning model:

- Get the data
- Clean the data
- Splitting the data into test and train sets
- Scaling the data
- Feature extraction and creation
- Choosing and fitting a model
- Cross-validation
- Scoring

With this number of steps, we are starting to risk mistakes - what if we accidentally mess up our splitting, and train on some of our validation data?

What happens when we want to change a step? What if we want to run it all at once every time? What if we want to carry out a parameter search on a step early in the process?

Luckily scikit-learn has a built in module, [pipeline](http://scikit-learn.org/stable/modules/pipeline.html) to take care of this for us.

We can build a pipeline that consists of any number of transformers and an estimator. When fitting the pipeline, each intermediate model calls `.fit()` and also `.transform()` on the data, passing this transformed data to the next model in the pipeline. However, the terminal model only calls `.fit()`.

Using a pipeline, we can make sure that all of our steps are carried out, with the same parameters, and in the same order.

### Machine Learning Process Example

Let's pretend we want to fit the iris dataset using a standard scaler, then a PCA and a SVM. NB, this is not an overly sensible pipeline, but it will allow us to run it in a reasonable time!

First, we can make it the standard way:


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#a bunch of imports!
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import svm

# Ignore futurewarnings
import warnings
warnings.filterwarnings('ignore')

# Load the iris data
iris = load_iris()

# Split the data into train &  test
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)

#scale
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

#decompose
pca = PCA(n_components=3)
pca.fit(X_train)
X_train = pca.transform(X_train)

#fit
rbf_svc = svm.SVC(kernel='rbf')
model = rbf_svc.fit(X_train, y_train)

print(model.score(X_train, y_train))

X_test = scaler.transform(X_test)
X_test = pca.transform(X_test)
model.score(X_test, y_test)
```

    0.9777777777777777





    0.9333333333333333



#### Predictions

Recall, we have to apply the same transforms to our test data: 


```python
# Carve out exactly the same train and test set
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)

X_test = pca.transform(X_test)
model.score(X_test, y_test)
```




    0.35



What happened? That's a terrible score!

We forgot a step - the scaler. We also needed to make sure everything is in the same order.


```python
# Carve out exactly the same train and test set
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)

X_test = scaler.transform(X_test)
X_test = pca.transform(X_test)
model.score(X_test, y_test)
```




    0.9333333333333333



### Pipeline

If we pipeline all our steps together, we don't have to remember.

The pipeline allows us to name steps, and put them together in a single object:


```python
from sklearn.pipeline import Pipeline

#we give our estimators as a list of tuples: name:function.
estimators = [('normalise', StandardScaler()),
              ('reduce_dim', PCA(n_components=3)),
              ('svm', svm.SVC(kernel='rbf'))]

pipe = Pipeline(estimators)
```

When calling the `fit` method on a pipeline, `.fit()` is called on each individual transformer, then `.transform()` with the output passed along to the next step in the pipeline, until reaching the final step (usually an estimator _i.e._ model).

This way we can automate our entire process - if we are carrying out a computationally complex data transformations and model fitting, we can set it all up, then call just call `.fit()` and go for lunch.

When calling `predict` on an already fitted pipeline, `.transform()` is called on each transformer in the pipeline, passing the output to the next step until finally calling `.predict()` on the final step (model). The same is true when using any scoring functions on a pipeline or calling it directly:

<img src="https://drive.google.com/uc?export=view&id=1o3tB_UXqts6Smk3jjP70kXGy6UMZZ6ql" width="75%">



```python
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)
    
pipe.fit(X_train, y_train)

pipe.score(X_test, y_test)
```




    0.9333333333333333



--- 

#### Exercise 1

Create a pipeline to fit a scaler, dimensional reduction and classifier to the Breast Cancer dataset. Carry it out on a train test split of the data. How does it look? Choose any combination of scalers, reducers and classifiers you want.

---


```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
```


```python
from sklearn.datasets import load_breast_cancer

# Load the cancer data
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=111)
```


```python
#we give our estimators as a list of tuples: name:function.
estimators_k = [('normalise', MinMaxScaler()),
              ('reduce_dim', PCA(n_components=5)),
              ('knn', KNeighborsRegressor(n_neighbors=10, weights='distance'))]

pipe_k = Pipeline(estimators)
```


```python
pipe_k.fit(X_train, y_train)

pipe_k.score(X_test, y_test)
```




    0.9736842105263158




```python
#we give our estimators as a list of tuples: name:function.
estimators_d = [('reduce_dim', PCA(n_components=3)),
              ('dt', DecisionTreeClassifier(max_depth=5))]

pipe_d = Pipeline(estimators)
```


```python
pipe_d.fit(X_train, y_train)

pipe_d.score(X_test, y_test)
```




    0.956140350877193




```python
#we give our estimators as a list of tuples: name:function.
estimators = [('normalise', MinMaxScaler()),
              ('reduce_dim', PCA(n_components=7)),
              ('svm', svm.SVC(kernel='linear'))]

pipe_s = Pipeline(estimators)
```


```python
pipe_s.fit(X_train, y_train)

pipe_s.score(X_test, y_test)
```




    0.9736842105263158



### Feature Union

Feature union is conceptually related to pipelining. Instead of putting together multiple steps in sequence, this allows us to do it in parallel, speeding up computation. We are in charge of making sure that the feature we construct are sensible! We can further put these inside pipelines...


```python
# Setting up some data. 
city_df = pd.DataFrame({'city': ['London', 'Toronto', 'Paris'],
                        'review': ['Super cool amazing', 'Very nice place', 'Its ok, good museums']})
```


```python
city_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city</th>
      <th>review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>London</td>
      <td>Super cool amazing</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Toronto</td>
      <td>Very nice place</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Paris</td>
      <td>Its ok, good museums</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Instantiate a list of tuples - each tuple is the name of the transform + the transformer
vectorizers = [('count_vect', CountVectorizer()), ('tfidf', TfidfVectorizer())]

# Create feature union
featunion = FeatureUnion(vectorizers)

# Fit and transform
vectorized = featunion.fit_transform(city_df['review'])

# View result
pd.DataFrame(vectorized.todense(), columns=featunion.get_feature_names())
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count_vect__amazing</th>
      <th>count_vect__cool</th>
      <th>count_vect__good</th>
      <th>count_vect__its</th>
      <th>count_vect__museums</th>
      <th>count_vect__nice</th>
      <th>count_vect__ok</th>
      <th>count_vect__place</th>
      <th>count_vect__super</th>
      <th>count_vect__very</th>
      <th>tfidf__amazing</th>
      <th>tfidf__cool</th>
      <th>tfidf__good</th>
      <th>tfidf__its</th>
      <th>tfidf__museums</th>
      <th>tfidf__nice</th>
      <th>tfidf__ok</th>
      <th>tfidf__place</th>
      <th>tfidf__super</th>
      <th>tfidf__very</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.57735</td>
      <td>0.57735</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.57735</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.57735</td>
      <td>0.0</td>
      <td>0.57735</td>
      <td>0.00000</td>
      <td>0.57735</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.00000</td>
      <td>0.5</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
    </tr>
  </tbody>
</table>
</div>



Here we can see the result is column-wise concatenation of the two transformations applied in parallel. Since there are no dependencies, the transformations can be applied in parallel and the results combined. This is a toy example which may or may not be practical. In practice, we'd often want to apply different types of feature engineering or data preprocessing tasks to different data types which may all be in the same DataFrame - for this there is a related and more advanced composite transformer which we investigate below.

### Column Transformer

The [ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html) is very similar to the Feature Union, only we can apply the transformers to specific columns or column subsets. This makes it very useful for when we have different data types that require different feature extraction or preprocessing techniques. Here we will take our city review data from the previous example and apply one-hot encoding on the city name, while simultaneously applying text vectorization on the review column:


```python
# Check
city_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city</th>
      <th>review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>London</td>
      <td>Super cool amazing</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Toronto</td>
      <td>Very nice place</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Paris</td>
      <td>Its ok, good museums</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Create the column transformations list + columns to which to apply
col_transforms = [('city_transform', OneHotEncoder(), ['city']),
                ('review_transform', TfidfVectorizer(), 'review')]

# Create the column transformer
col_trans = ColumnTransformer(col_transforms)

# Fit
col_trans.fit(city_df)
```




    ColumnTransformer(transformers=[('city_transform', OneHotEncoder(), ['city']),
                                    ('review_transform', TfidfVectorizer(),
                                     'review')])




```python
# to combine above iwht something on all data: 
#estimators =((col_Trans .... ,
 #            scale thing... on all ))
```

The new feature names are contained in the fitted transformer as we saw before with FeatureUnion:


```python
# Feature names
col_trans.get_feature_names()
```




    ['city_transform__x0_London',
     'city_transform__x0_Paris',
     'city_transform__x0_Toronto',
     'review_transform__amazing',
     'review_transform__cool',
     'review_transform__good',
     'review_transform__its',
     'review_transform__museums',
     'review_transform__nice',
     'review_transform__ok',
     'review_transform__place',
     'review_transform__super',
     'review_transform__very']



Now we can apply the transformation and create a new DataFrame:


```python
# Apply the transformations
transformed = col_trans.transform(city_df) 

# Check
transformed
```




    array([[1.        , 0.        , 0.        , 0.57735027, 0.57735027,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.57735027, 0.        ],
           [0.        , 0.        , 1.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.57735027, 0.        ,
            0.57735027, 0.        , 0.57735027],
           [0.        , 1.        , 0.        , 0.        , 0.        ,
            0.5       , 0.5       , 0.5       , 0.        , 0.5       ,
            0.        , 0.        , 0.        ]])




```python
# Put in a DataFrame
transformed_df = pd.DataFrame(transformed, columns=col_trans.get_feature_names())
transformed_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city_transform__x0_London</th>
      <th>city_transform__x0_Paris</th>
      <th>city_transform__x0_Toronto</th>
      <th>review_transform__amazing</th>
      <th>review_transform__cool</th>
      <th>review_transform__good</th>
      <th>review_transform__its</th>
      <th>review_transform__museums</th>
      <th>review_transform__nice</th>
      <th>review_transform__ok</th>
      <th>review_transform__place</th>
      <th>review_transform__super</th>
      <th>review_transform__very</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.57735</td>
      <td>0.57735</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.57735</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.57735</td>
      <td>0.0</td>
      <td>0.57735</td>
      <td>0.00000</td>
      <td>0.57735</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.00000</td>
      <td>0.5</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Check
pd.concat([city_df, transformed_df], axis=1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city</th>
      <th>review</th>
      <th>city_transform__x0_London</th>
      <th>city_transform__x0_Paris</th>
      <th>city_transform__x0_Toronto</th>
      <th>review_transform__amazing</th>
      <th>review_transform__cool</th>
      <th>review_transform__good</th>
      <th>review_transform__its</th>
      <th>review_transform__museums</th>
      <th>review_transform__nice</th>
      <th>review_transform__ok</th>
      <th>review_transform__place</th>
      <th>review_transform__super</th>
      <th>review_transform__very</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>London</td>
      <td>Super cool amazing</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.57735</td>
      <td>0.57735</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.57735</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Toronto</td>
      <td>Very nice place</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.57735</td>
      <td>0.0</td>
      <td>0.57735</td>
      <td>0.00000</td>
      <td>0.57735</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Paris</td>
      <td>Its ok, good museums</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.00000</td>
      <td>0.5</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
    </tr>
  </tbody>
</table>
</div>




```python
pipe[1]
```




    PCA(n_components=3)




```python
pipe.named_steps
```




    {'normalise': StandardScaler(),
     'reduce_dim': PCA(n_components=3),
     'svm': SVC()}



If we did this without the column transformer we would have to create seperate OHE and count vectorizer objects, fit on our training data and then transform it and the test data. Then to bring the two arrays back together we would have to manually concatenate with `np.hstack` or convert to DataFrames and use `pd.concat` with relevant column headings. As we saw above, we can instead just set up the column transformer to do it all for us, all in one go!

Just like with feature union, you can pass the column transformer into a sklearn pipeline for reproducible complex data preprocessing to apply to train and test sets, in cross-validation, or to new data to score against.

### Leakage

Something we can accidentally do is "*leak*" information from our test or validation set into our model.

Recall from feature selection that we can choose certain variables based on how well they predict the data, and include or exclude them.

In the naïve case, if we fit this on the entire data, and then carry out a cross-validation, we will score very well! The problem is that we are validating on data we have already seen.

Let's start with some random data:


```python
X = np.random.normal(size=(100, 10000))
y = np.random.normal(size=(100,))
```


```python

```

There should be no relation between X and y in this data! We can guess there will be some subsets of columns that will match well, but if we properly cross-validate and do feature selection, we would remove them.

See the example below - we have carried out feature selection using the `SelectPercentile` method. This is very similar to the `SelectKBest` technique: `SelectPercentile` scores each feature separately based on how good it predicts the target variable and only keeps those features that scored in a user-specified top percentile. We used the top 5th percentile which corresponds to keeping 5% of the original columns.


```python
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge

# split for testing
X_remain, X_test, y_remain, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

select = SelectPercentile(score_func=f_regression, percentile=5).fit(X_remain, y_remain)
X_remain_selected = select.transform(X_remain)

cross_val_score(Ridge(), X_remain_selected, y_remain, cv=5)
```




    array([0.86481209, 0.95121555, 0.93682488, 0.9332601 , 0.90357904])



We made a very bad mistake! We have fit a model that we are claiming has an $R^2$ score of ~0.90 on completely random data. This is purely the result of leakage. If we now test such a model on unseen data, the performance will fall down to ~0.0 so our CV-score is a completely useless proxy for performance.


```python
# setup the model as before
model = Ridge()
model.fit(X_remain_selected, y_remain)

# test it on truly unseen data
X_test_selected = select.transform(X_test)
model.score(X_test_selected, y_test)
```




    -0.11741950602469942



Let's correct our mistake and put the feature selection and regression into a pipeline:


```python
pipe = Pipeline([("select", SelectPercentile(score_func=f_regression, percentile=5)),
                 ("ridge", Ridge())])

cross_val_score(pipe, X, y, cv=5)
```




    array([-0.12450477,  0.00270798, -0.12460739, -0.40459538, -0.24078813])



Now we have a more sensible prediction: a score of almost zero. We have chained together the feature selection with the cross-validation, so that at each step we pass in the subsetted X data. This ensures that the cross validation is a good estimate of the model performance on unseen data and we are not overfitting.

In addition to feature selection, pipelines allow the proper application of scaling in cross-validation without leakage. Parameter estimates (*e.g.* mean, variance) should be calculated from the training data only in each CV fold:


```python
# First do a train-test split
X_remain, X_test, y_remain, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# INCORRECT - scale the whole data set, then do cross-val on the scaled data
my_ss = StandardScaler()
X_scaled = my_ss.fit_transform(X_remain)

# Score
incorrect_score = cross_val_score(Ridge(), X_scaled, y_remain)
print(incorrect_score)
# CORRECT - create a pipeline with a scaler, such that for each fold, it is fit only on the training set
my_pipeline = Pipeline([('ss', StandardScaler()), ('ridge', Ridge())])

# Score
correct_score = cross_val_score(my_pipeline, X_remain, y_remain)
print(correct_score)
```

    [-0.07753297 -0.52846359 -0.05843659 -0.04788618 -0.43564178]
    [-0.07124587 -0.53035688 -0.06171549 -0.05559539 -0.43666715]


### Grid Searching

We can search over multiple parameters in a model, or pipeline by using [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).

In this way, we can define a pipeline, and then exhaustively test all possible combinations of hyperparameters to see which will give us the best score on a given metrics. Let's redefine our pipeline from earlier, with empty parameters:


```python
estimators = [('normalise', StandardScaler()),
              ('reduce_dim', PCA()),
              ('svm', svm.SVC())]
pipe = Pipeline(estimators)
```

We want to choose hyperparameters for our PCA (number of components), and SVM (kernel type).

To do that, we need to pass in the parameters to GridSearchCV as a dict, in the format `stepname__param: 
[values]`:


```python
from sklearn.model_selection import GridSearchCV

params = {'svm__kernel': ['linear','rbf'], 
          'reduce_dim__n_components': [1, 2, 3]}

grid_search = GridSearchCV(pipe, param_grid=params)
```

We now have a grid search object with which we can fit to our data, and find the values for multiple hyperparameters. For our example, our "grid" to search over is in 2-D with only 6 possible combinations:

<img src="https://drive.google.com/uc?export=view&id=1NNs4ulltJupGN7kzse0-9W1Pzik9IyXh" width=75%>

The problem is, we are doing an exhaustive search over all possible combinations of hyperparameters - here we only have 2 by 3 possible combinations, which is manageable - but as we try and increase the number to test it will take a long time, especially if we have expensive models to build; the number of possible points in the grid is multiplicative and the grid to search gains and additional dimension for each set of possible hyperparameter values to explore. As such, generally speaking doing grid search is a computationally expensive operation, as we will see below.

We can pass in the `memory` argument to our pipeline, so that we can cache results on the same object (you can see this helps a lot as we increase the number of steps). For this example, we are not too worried, however we will show demonstrative usage below:


```python
from tempfile import mkdtemp
cachedir = mkdtemp()
pipe = Pipeline(estimators, memory = cachedir)

params = {'svm__kernel': ['linear','rbf'], 
          'reduce_dim__n_components': [1, 2, 3]}

grid_search = GridSearchCV(pipe, param_grid=params) #, verbose=2)

X_train, X_test, y_train, y_test =\
   train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)
    
fitted_search = grid_search.fit(X_train, y_train)
```

The fitted search is the best found model, based on its cross-validation score in the grid search.

We can predict and score using the object:


```python
fitted_search.score(X_test, y_test)
```




    0.9333333333333333



And get out the best parameters:


```python
fitted_search.best_estimator_
```




    Pipeline(memory='/var/folders/fj/hxsz7nhs25b5xftsj_gmk99c0000gn/T/tmpkuow39zu',
             steps=[('normalise', StandardScaler()),
                    ('reduce_dim', PCA(n_components=3)),
                    ('svm', SVC(kernel='linear'))])



And all the scores for each CV fold:


```python
fitted_search.cv_results_['mean_test_score']
```




    array([0.92222222, 0.92222222, 0.91111111, 0.91111111, 0.97777778,
           0.96666667])




```python
fitted_search.
```




    <function sklearn.metrics._scorer._passthrough_scorer(estimator, *args, **kwargs)>



As well as choosing the best parameters, we can also choose between models, and whether to even carry a step out:


```python
from sklearn.tree import DecisionTreeClassifier

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

estimators = [('normalise', StandardScaler()),
              ('model', svm.SVC())]

pipe = Pipeline(estimators)

param_grid = [
            {'model': [svm.SVC()], 
             'normalise': [StandardScaler(), None],
             'model__gamma': [0.001, 0.01, 0.1, 1, 10, 100],
             'model__C': [0.001, 0.01, 0.1, 1, 10, 100]},
            {'model': [DecisionTreeClassifier()],
             'normalise': [None], 'model__max_depth': [1, 2, 3]}
]

grid = GridSearchCV(pipe, param_grid, cv=5)
fittedgrid = grid.fit(X_train, y_train)
```


```python
# Best estimator object
fittedgrid.best_estimator_
```




    Pipeline(steps=[('normalise', None), ('model', SVC(C=10, gamma=0.1))])




```python
# Best hyperparameters
fittedgrid.best_params_
```




    {'model': SVC(C=10, gamma=0.1),
     'model__C': 10,
     'model__gamma': 0.1,
     'normalise': None}




```python
# Mean test score for each CV fold
fittedgrid.cv_results_['mean_test_score']
```




    array([0.37777778, 0.37777778, 0.37777778, 0.37777778, 0.37777778,
           0.37777778, 0.37777778, 0.37777778, 0.37777778, 0.37777778,
           0.37777778, 0.37777778, 0.37777778, 0.37777778, 0.37777778,
           0.37777778, 0.37777778, 0.37777778, 0.37777778, 0.37777778,
           0.37777778, 0.37777778, 0.37777778, 0.37777778, 0.37777778,
           0.37777778, 0.37777778, 0.53333333, 0.86666667, 0.95555556,
           0.88888889, 0.96666667, 0.37777778, 0.37777778, 0.37777778,
           0.37777778, 0.37777778, 0.65555556, 0.92222222, 0.95555556,
           0.96666667, 0.97777778, 0.96666667, 0.96666667, 0.91111111,
           0.94444444, 0.4       , 0.44444444, 0.92222222, 0.95555556,
           0.96666667, 0.97777778, 0.97777778, 0.98888889, 0.95555556,
           0.97777778, 0.91111111, 0.94444444, 0.42222222, 0.45555556,
           0.96666667, 0.97777778, 0.97777778, 0.98888889, 0.93333333,
           0.94444444, 0.95555556, 0.96666667, 0.91111111, 0.94444444,
           0.42222222, 0.45555556, 0.7       , 0.95555556, 0.95555556])



By default, GridSearchCV will perform cross-validation to find the best hyperparameters for the specified metric, then as a final step, re-fit with the best combination on the entire dataset. This final model is returned in `best_estimator_`. (This behavior can be toggled with use of the `refit` boolean parameter if desired.)

---

#### Exercise 2

1. Create and fit a `GridSearchCV` model on the breast cancer dataset - use at least two scalers (counting None), dimensionality reduction, and three classifiers. Use at least three values each for the hyperparameters for the dimensionality reduction and models.

2. Do you think grid search would give the same result if we changed the data set?

---


```python
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# load in data
breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=111)
```


```python
estimators = [('normalise', StandardScaler()),
              ('dim_reducer', PCA()),
              ('model', svm.SVC())]

pipe = Pipeline(estimators)

param_grid = [
            {'model': [svm.SVC()], 
             'normalise': [StandardScaler(), MinMaxScaler(), None],
             'dim_reducer': [PCA(), None],
             'model__gamma': [0.011, 0.01, 0.009, 0.012, 0.008, 0.013],
             'model__C': [5.5, 1, 10, 6.5, 6.7, 6.3]},
            {'model': [DecisionTreeClassifier()],
             'normalise': [None],
             'dim_reducer': [PCA(), None],
             'model__max_depth': [1, 2, 3]},
            {'model': [LogisticRegression()],
             'normalise': [StandardScaler(), MinMaxScaler(), None],
             'dim_reducer': [PCA(), None],
             'model__penalty': ['l1', 'l2'],
             'model__C': [10**x for x in range(-3,3)]}
]

grid = GridSearchCV(pipe, param_grid, cv=5, verbose=1)
fittedgrid = grid.fit(X_train, y_train)
```

    Fitting 5 folds for each of 294 candidates, totalling 1470 fits


dimensionality reduction, and three classifiers. Use at least three values each for the hyperparameters for the dimensionality reduction and models.



```python
# Best estimator object
fittedgrid.best_estimator_
```




    Pipeline(steps=[('normalise', StandardScaler()), ('dim_reducer', PCA()),
                    ('model', SVC(C=5.5, gamma=0.008))])




```python
# Best hyperparameters
fittedgrid.best_params_
```




    {'dim_reducer': PCA(),
     'model': SVC(C=5.5, gamma=0.008),
     'model__C': 5.5,
     'model__gamma': 0.008,
     'normalise': StandardScaler()}




```python
fittedgrid.score(X_test, y_test)
```




    0.9824561403508771




```python
# Mean test score for each CV fold
fittedgrid.cv_results_['mean_test_score']
```




    array([0.97582418, 0.96483516, 0.62637363, 0.97802198, 0.95604396,
           0.62857143, 0.97582418, 0.96923077, 0.62637363, 0.97362637,
           0.96703297, 0.62637363, 0.96923077, 0.96923077, 0.62637363,
           0.62637363, 0.62637363, 0.62637363, 0.97582418, 0.96703297,
           0.62637363, 0.97802198, 0.95824176, 0.62857143, 0.97362637,
           0.97142857, 0.62637363, 0.97142857, 0.96923077, 0.62637363,
           0.96263736, 0.97142857, 0.62637363, 0.62637363, 0.62637363,
           0.62637363, 0.97582418, 0.96703297, 0.62637363, 0.97802198,
           0.96043956, 0.62857143, 0.97362637, 0.96703297, 0.62637363,
           0.97142857, 0.97142857, 0.62637363, 0.96263736, 0.97362637,
           0.62637363, 0.62637363, 0.62637363, 0.62637363, 0.97362637,
           0.96923077, 0.62637363, 0.97802198, 0.96263736, 0.62857143,
           0.97362637, 0.96703297, 0.62637363, 0.96923077, 0.97142857,
           0.62637363, 0.96483516, 0.97142857, 0.62637363, 0.62637363,
           0.62637363, 0.62637363, 0.97582418, 0.97142857, 0.62637363,
           0.97802198, 0.96703297, 0.62857143, 0.97142857, 0.96923077,
           0.62637363, 0.96483516, 0.97362637, 0.62637363, 0.96263736,
           0.97142857, 0.62637363, 0.62637363, 0.62637363, 0.62637363,
           0.97582418, 0.96923077, 0.62637363, 0.97582418, 0.96483516,
           0.62857143, 0.97142857, 0.97142857, 0.62637363, 0.96703297,
           0.97142857, 0.62637363, 0.96263736, 0.96923077, 0.62637363,
           0.62637363, 0.62637363, 0.62637363, 0.90989011, 0.92307692,
           0.93846154, 0.87032967, 0.62637363, 0.93626374, 0.95164835,
           0.73846154, 0.94065934, 0.97142857, 0.92967033, 0.94065934,
           0.97582418, 0.96483516, 0.95164835, 0.96923077, 0.97142857,
           0.95164835, 0.96703297, 0.97582418, 0.94285714])



#### Conclusion

As we have seen, the ability to combine multiple transformers and estimators in pipelines (and other composite estimator types) allows automating many different data transformations and models: either in parallel, in series, or both. Together with grid search, we can also automate finding both the best combination of steps to apply as well as the optimal hyperparameters for a given model (or array of models).

However, doing so is computationally expensive and can be very time-consuming - for sizeable datasets, it is not unusual for grid searches to run for many hours, or even longer if the grid resolution is fine and hyperparameter space large. In practice, grid search is not a replacement for the manual exploration of model performance across different hyperparameter values; a data scientist should explore ranges of possible values coarsely to get an idea of which model types and hyperparameter values work well, then apply grid search in a more exacting manner in order to arrive the best model for a given problem.

Additionally, [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) can also be used, where the criteria of arriving at the guaranteed optimal combination of hyperparameters is loosened and only a subset of possible values are explored, in order to balance computation time with model performance.

<div id="container" style="position:relative;">
<div style="position:relative; float:right"><img style="height:25px""width: 50px" src ="https://drive.google.com/uc?export=view&id=14VoXUJftgptWtdNhtNYVm6cjVmEWpki1" />
</div>
</div>

