```python
import pandas as pd
import numpy as np
```


```python
data = {
    'first': ['John', 'Bob', 'Eve', 'Charlie', None, np.nan, 'Sophia', 'NA', 'MISSING'],
    'last': ['Doe', 'Johnson', 'Anderson', 'Brown', 'Miller', 'Wilson', 'Davis', 'Martinez', 'OKAY'],
    'email': ['john.doe@example.com', 'NA', None, 'charlie.brown@example.com',
              'grace.miller@example.com', 'will@gmail.com', np.nan, 'liam.martinez@example.com', 'MISSING'],
    'age': [25, np.nan, None, 40, 28, 45, np.nan, 37, 'MISSING'],
    'dob': ['1998-03-15', '1980-05-10', '1975-09-07', '1995-08-18', '1995-08-18', '1989-06-25',
            'NA', 'NA', 'MISSING']
}

df = pd.DataFrame(data)
df
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
      <th>first</th>
      <th>last</th>
      <th>email</th>
      <th>age</th>
      <th>dob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>John</td>
      <td>Doe</td>
      <td>john.doe@example.com</td>
      <td>25</td>
      <td>1998-03-15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bob</td>
      <td>Johnson</td>
      <td>NA</td>
      <td>NaN</td>
      <td>1980-05-10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Eve</td>
      <td>Anderson</td>
      <td>None</td>
      <td>None</td>
      <td>1975-09-07</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Charlie</td>
      <td>Brown</td>
      <td>charlie.brown@example.com</td>
      <td>40</td>
      <td>1995-08-18</td>
    </tr>
    <tr>
      <th>4</th>
      <td>None</td>
      <td>Miller</td>
      <td>grace.miller@example.com</td>
      <td>28</td>
      <td>1995-08-18</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NaN</td>
      <td>Wilson</td>
      <td>will@gmail.com</td>
      <td>45</td>
      <td>1989-06-25</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Sophia</td>
      <td>Davis</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NA</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NA</td>
      <td>Martinez</td>
      <td>liam.martinez@example.com</td>
      <td>37</td>
      <td>NA</td>
    </tr>
    <tr>
      <th>8</th>
      <td>MISSING</td>
      <td>OKAY</td>
      <td>MISSING</td>
      <td>MISSING</td>
      <td>MISSING</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Convert 'NA' and 'MISSING' to actual NaN values
df = df.replace(['NA', 'MISSING'], np.nan)
df

# df.replace('NA', np.nan, inplace=True)
# df.replace('MISSING', np.nan, inplace=True)
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
      <th>first</th>
      <th>last</th>
      <th>email</th>
      <th>age</th>
      <th>dob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>John</td>
      <td>Doe</td>
      <td>john.doe@example.com</td>
      <td>25.0</td>
      <td>1998-03-15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bob</td>
      <td>Johnson</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1980-05-10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Eve</td>
      <td>Anderson</td>
      <td>None</td>
      <td>NaN</td>
      <td>1975-09-07</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Charlie</td>
      <td>Brown</td>
      <td>charlie.brown@example.com</td>
      <td>40.0</td>
      <td>1995-08-18</td>
    </tr>
    <tr>
      <th>4</th>
      <td>None</td>
      <td>Miller</td>
      <td>grace.miller@example.com</td>
      <td>28.0</td>
      <td>1995-08-18</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NaN</td>
      <td>Wilson</td>
      <td>will@gmail.com</td>
      <td>45.0</td>
      <td>1989-06-25</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Sophia</td>
      <td>Davis</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NaN</td>
      <td>Martinez</td>
      <td>liam.martinez@example.com</td>
      <td>37.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>NaN</td>
      <td>OKAY</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Detect missing values.
df.isna()
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
      <th>first</th>
      <th>last</th>
      <th>email</th>
      <th>age</th>
      <th>dob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>8</th>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Fill NA/NaN values using the specified method.
df.fillna("<MISSING>")
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
      <th>first</th>
      <th>last</th>
      <th>email</th>
      <th>age</th>
      <th>dob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>John</td>
      <td>Doe</td>
      <td>john.doe@example.com</td>
      <td>25.0</td>
      <td>1998-03-15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bob</td>
      <td>Johnson</td>
      <td>&lt;MISSING&gt;</td>
      <td>&lt;MISSING&gt;</td>
      <td>1980-05-10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Eve</td>
      <td>Anderson</td>
      <td>&lt;MISSING&gt;</td>
      <td>&lt;MISSING&gt;</td>
      <td>1975-09-07</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Charlie</td>
      <td>Brown</td>
      <td>charlie.brown@example.com</td>
      <td>40.0</td>
      <td>1995-08-18</td>
    </tr>
    <tr>
      <th>4</th>
      <td>&lt;MISSING&gt;</td>
      <td>Miller</td>
      <td>grace.miller@example.com</td>
      <td>28.0</td>
      <td>1995-08-18</td>
    </tr>
    <tr>
      <th>5</th>
      <td>&lt;MISSING&gt;</td>
      <td>Wilson</td>
      <td>will@gmail.com</td>
      <td>45.0</td>
      <td>1989-06-25</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Sophia</td>
      <td>Davis</td>
      <td>&lt;MISSING&gt;</td>
      <td>&lt;MISSING&gt;</td>
      <td>&lt;MISSING&gt;</td>
    </tr>
    <tr>
      <th>7</th>
      <td>&lt;MISSING&gt;</td>
      <td>Martinez</td>
      <td>liam.martinez@example.com</td>
      <td>37.0</td>
      <td>&lt;MISSING&gt;</td>
    </tr>
    <tr>
      <th>8</th>
      <td>&lt;MISSING&gt;</td>
      <td>OKAY</td>
      <td>&lt;MISSING&gt;</td>
      <td>&lt;MISSING&gt;</td>
      <td>&lt;MISSING&gt;</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Remove row's missing values 
df.dropna() # axis=0/index
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
      <th>first</th>
      <th>last</th>
      <th>email</th>
      <th>age</th>
      <th>dob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>John</td>
      <td>Doe</td>
      <td>john.doe@example.com</td>
      <td>25.0</td>
      <td>1998-03-15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Charlie</td>
      <td>Brown</td>
      <td>charlie.brown@example.com</td>
      <td>40.0</td>
      <td>1995-08-18</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Remove column if there is single missing value 
df.dropna(axis='columns')
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
      <th>last</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Doe</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Johnson</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Anderson</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Brown</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Miller</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Wilson</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Davis</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Martinez</td>
    </tr>
    <tr>
      <th>8</th>
      <td>OKAY</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Exercise: Drop all rows that contain at least one missing value
df.dropna(axis='index', how='any')
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
      <th>first</th>
      <th>last</th>
      <th>email</th>
      <th>age</th>
      <th>dob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>John</td>
      <td>Doe</td>
      <td>john.doe@example.com</td>
      <td>25.0</td>
      <td>1998-03-15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Charlie</td>
      <td>Brown</td>
      <td>charlie.brown@example.com</td>
      <td>40.0</td>
      <td>1995-08-18</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Exercise: Drop only the rows where column 'Age' has missing values
df.dropna(subset=['email'])
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
      <th>first</th>
      <th>last</th>
      <th>email</th>
      <th>age</th>
      <th>dob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>John</td>
      <td>Doe</td>
      <td>john.doe@example.com</td>
      <td>25.0</td>
      <td>1998-03-15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Charlie</td>
      <td>Brown</td>
      <td>charlie.brown@example.com</td>
      <td>40.0</td>
      <td>1995-08-18</td>
    </tr>
    <tr>
      <th>4</th>
      <td>None</td>
      <td>Miller</td>
      <td>grace.miller@example.com</td>
      <td>28.0</td>
      <td>1995-08-18</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NaN</td>
      <td>Wilson</td>
      <td>will@gmail.com</td>
      <td>45.0</td>
      <td>1989-06-25</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NaN</td>
      <td>Martinez</td>
      <td>liam.martinez@example.com</td>
      <td>37.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df
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
      <th>first</th>
      <th>last</th>
      <th>email</th>
      <th>age</th>
      <th>dob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>John</td>
      <td>Doe</td>
      <td>john.doe@example.com</td>
      <td>25.0</td>
      <td>1998-03-15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bob</td>
      <td>Johnson</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1980-05-10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Eve</td>
      <td>Anderson</td>
      <td>None</td>
      <td>NaN</td>
      <td>1975-09-07</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Charlie</td>
      <td>Brown</td>
      <td>charlie.brown@example.com</td>
      <td>40.0</td>
      <td>1995-08-18</td>
    </tr>
    <tr>
      <th>4</th>
      <td>None</td>
      <td>Miller</td>
      <td>grace.miller@example.com</td>
      <td>28.0</td>
      <td>1995-08-18</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NaN</td>
      <td>Wilson</td>
      <td>will@gmail.com</td>
      <td>45.0</td>
      <td>1989-06-25</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Sophia</td>
      <td>Davis</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NaN</td>
      <td>Martinez</td>
      <td>liam.martinez@example.com</td>
      <td>37.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>NaN</td>
      <td>OKAY</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Exercise:  Drop rows with more than Four missing values
df.dropna(thresh=4)
# non-missing values required to keep a row or column.
# If the number of non-missing/valid values is below the specified threshold, the row or column is dropped.
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
      <th>first</th>
      <th>last</th>
      <th>email</th>
      <th>age</th>
      <th>dob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>John</td>
      <td>Doe</td>
      <td>john.doe@example.com</td>
      <td>25.0</td>
      <td>1998-03-15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Charlie</td>
      <td>Brown</td>
      <td>charlie.brown@example.com</td>
      <td>40.0</td>
      <td>1995-08-18</td>
    </tr>
    <tr>
      <th>4</th>
      <td>None</td>
      <td>Miller</td>
      <td>grace.miller@example.com</td>
      <td>28.0</td>
      <td>1995-08-18</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NaN</td>
      <td>Wilson</td>
      <td>will@gmail.com</td>
      <td>45.0</td>
      <td>1989-06-25</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(df) 
df.dropna(how='any') # Drop the row if any of its elements are missing.
```

         first      last                      email   age         dob
    0     John       Doe       john.doe@example.com  25.0  1998-03-15
    1      Bob   Johnson                        NaN   NaN  1980-05-10
    2      Eve  Anderson                       None   NaN  1975-09-07
    3  Charlie     Brown  charlie.brown@example.com  40.0  1995-08-18
    4     None    Miller   grace.miller@example.com  28.0  1995-08-18
    5      NaN    Wilson             will@gmail.com  45.0  1989-06-25
    6   Sophia     Davis                        NaN   NaN         NaN
    7      NaN  Martinez  liam.martinez@example.com  37.0         NaN
    8      NaN      OKAY                        NaN   NaN         NaN





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
      <th>first</th>
      <th>last</th>
      <th>email</th>
      <th>age</th>
      <th>dob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>John</td>
      <td>Doe</td>
      <td>john.doe@example.com</td>
      <td>25.0</td>
      <td>1998-03-15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Charlie</td>
      <td>Brown</td>
      <td>charlie.brown@example.com</td>
      <td>40.0</td>
      <td>1995-08-18</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.replace('OKAY', np.nan, inplace=True)
print(df)
df.dropna(how='all') # Drop the row only if all of its elements are missing.
```

         first      last                      email   age         dob
    0     John       Doe       john.doe@example.com  25.0  1998-03-15
    1      Bob   Johnson                        NaN   NaN  1980-05-10
    2      Eve  Anderson                       None   NaN  1975-09-07
    3  Charlie     Brown  charlie.brown@example.com  40.0  1995-08-18
    4     None    Miller   grace.miller@example.com  28.0  1995-08-18
    5      NaN    Wilson             will@gmail.com  45.0  1989-06-25
    6   Sophia     Davis                        NaN   NaN         NaN
    7      NaN  Martinez  liam.martinez@example.com  37.0         NaN
    8      NaN       NaN                        NaN   NaN         NaN





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
      <th>first</th>
      <th>last</th>
      <th>email</th>
      <th>age</th>
      <th>dob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>John</td>
      <td>Doe</td>
      <td>john.doe@example.com</td>
      <td>25.0</td>
      <td>1998-03-15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bob</td>
      <td>Johnson</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1980-05-10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Eve</td>
      <td>Anderson</td>
      <td>None</td>
      <td>NaN</td>
      <td>1975-09-07</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Charlie</td>
      <td>Brown</td>
      <td>charlie.brown@example.com</td>
      <td>40.0</td>
      <td>1995-08-18</td>
    </tr>
    <tr>
      <th>4</th>
      <td>None</td>
      <td>Miller</td>
      <td>grace.miller@example.com</td>
      <td>28.0</td>
      <td>1995-08-18</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NaN</td>
      <td>Wilson</td>
      <td>will@gmail.com</td>
      <td>45.0</td>
      <td>1989-06-25</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Sophia</td>
      <td>Davis</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NaN</td>
      <td>Martinez</td>
      <td>liam.martinez@example.com</td>
      <td>37.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Exercise: Drop rows where both 'email' and 'age' are missing with columns 'email' and 'age',
df.dropna(axis='index', how='all', subset=['email', 'age'])
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
      <th>first</th>
      <th>last</th>
      <th>email</th>
      <th>age</th>
      <th>dob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>John</td>
      <td>Doe</td>
      <td>john.doe@example.com</td>
      <td>25.0</td>
      <td>1998-03-15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Charlie</td>
      <td>Brown</td>
      <td>charlie.brown@example.com</td>
      <td>40.0</td>
      <td>1995-08-18</td>
    </tr>
    <tr>
      <th>4</th>
      <td>None</td>
      <td>Miller</td>
      <td>grace.miller@example.com</td>
      <td>28.0</td>
      <td>1995-08-18</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NaN</td>
      <td>Wilson</td>
      <td>will@gmail.com</td>
      <td>45.0</td>
      <td>1989-06-25</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NaN</td>
      <td>Martinez</td>
      <td>liam.martinez@example.com</td>
      <td>37.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Exercise: Fill missing values in 'age' column with the mean age

df_filled_age = df.copy()
df_filled_age['age'] = pd.to_numeric(df_filled_age['age'], errors='coerce')  # Convert to numeric
mean_age = df_filled_age['age'].mean()
df_filled_age['age'].fillna(mean_age, inplace=True)

print("\nExercise 3: DataFrame after filling missing values in 'age' column with the mean age:")
df_filled_age

```

    
    Exercise 3: DataFrame after filling missing values in 'age' column with the mean age:





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
      <th>first</th>
      <th>last</th>
      <th>email</th>
      <th>age</th>
      <th>dob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>John</td>
      <td>Doe</td>
      <td>john.doe@example.com</td>
      <td>25.0</td>
      <td>1998-03-15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bob</td>
      <td>Johnson</td>
      <td>NaN</td>
      <td>35.0</td>
      <td>1980-05-10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Eve</td>
      <td>Anderson</td>
      <td>None</td>
      <td>35.0</td>
      <td>1975-09-07</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Charlie</td>
      <td>Brown</td>
      <td>charlie.brown@example.com</td>
      <td>40.0</td>
      <td>1995-08-18</td>
    </tr>
    <tr>
      <th>4</th>
      <td>None</td>
      <td>Miller</td>
      <td>grace.miller@example.com</td>
      <td>28.0</td>
      <td>1995-08-18</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NaN</td>
      <td>Wilson</td>
      <td>will@gmail.com</td>
      <td>45.0</td>
      <td>1989-06-25</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Sophia</td>
      <td>Davis</td>
      <td>NaN</td>
      <td>35.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NaN</td>
      <td>Martinez</td>
      <td>liam.martinez@example.com</td>
      <td>37.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>35.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.dtypes
```




    first     object
    last      object
    email     object
    age      float64
    dob       object
    dtype: object




```python
# Exercise: Cast age column as a float data type.
df['age'] = df['age'].astype(float)
df
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
      <th>first</th>
      <th>last</th>
      <th>email</th>
      <th>age</th>
      <th>dob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>John</td>
      <td>Doe</td>
      <td>john.doe@example.com</td>
      <td>25.0</td>
      <td>1998-03-15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bob</td>
      <td>Johnson</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1980-05-10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Eve</td>
      <td>Anderson</td>
      <td>None</td>
      <td>NaN</td>
      <td>1975-09-07</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Charlie</td>
      <td>Brown</td>
      <td>charlie.brown@example.com</td>
      <td>40.0</td>
      <td>1995-08-18</td>
    </tr>
    <tr>
      <th>4</th>
      <td>None</td>
      <td>Miller</td>
      <td>grace.miller@example.com</td>
      <td>28.0</td>
      <td>1995-08-18</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NaN</td>
      <td>Wilson</td>
      <td>will@gmail.com</td>
      <td>45.0</td>
      <td>1989-06-25</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Sophia</td>
      <td>Davis</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NaN</td>
      <td>Martinez</td>
      <td>liam.martinez@example.com</td>
      <td>37.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


