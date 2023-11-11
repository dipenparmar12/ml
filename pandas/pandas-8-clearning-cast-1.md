```python
import pandas as pd
import numpy as np
```


```python
df = pd.read_csv('./output_minified.csv')
schema = pd.read_csv('./stack-overflow-survey-2023/survey_results_schema.csv')

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
      <th>id</th>
      <th>ConvertedCompYearly</th>
      <th>Country</th>
      <th>Currency</th>
      <th>ConvertedCompYearly.1</th>
      <th>LanguageHaveWorkedWith</th>
      <th>RemoteWork</th>
      <th>CompTotal</th>
      <th>WorkExp</th>
      <th>SOAccount</th>
      <th>Employment</th>
      <th>YearsCode</th>
      <th>YearsCodePro</th>
      <th>DevType</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>285000.0</td>
      <td>United States of America</td>
      <td>USD\tUnited States dollar</td>
      <td>285000.0</td>
      <td>HTML/CSS;JavaScript;Python</td>
      <td>Remote</td>
      <td>2.850000e+05</td>
      <td>10.0</td>
      <td>Yes</td>
      <td>Employed, full-time</td>
      <td>18</td>
      <td>9</td>
      <td>Senior Executive (C-Suite, VP, etc.)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>250000.0</td>
      <td>United States of America</td>
      <td>USD\tUnited States dollar</td>
      <td>250000.0</td>
      <td>Bash/Shell (all shells);Go</td>
      <td>Hybrid (some remote, some in-person)</td>
      <td>2.500000e+05</td>
      <td>23.0</td>
      <td>Yes</td>
      <td>Employed, full-time</td>
      <td>27</td>
      <td>23</td>
      <td>Developer, back-end</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>156000.0</td>
      <td>United States of America</td>
      <td>USD\tUnited States dollar</td>
      <td>156000.0</td>
      <td>Bash/Shell (all shells);HTML/CSS;JavaScript;PH...</td>
      <td>Hybrid (some remote, some in-person)</td>
      <td>1.560000e+05</td>
      <td>7.0</td>
      <td>Yes</td>
      <td>Employed, full-time</td>
      <td>12</td>
      <td>7</td>
      <td>Developer, front-end</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>23456.0</td>
      <td>Philippines</td>
      <td>PHP\tPhilippine peso</td>
      <td>23456.0</td>
      <td>HTML/CSS;JavaScript;TypeScript</td>
      <td>Remote</td>
      <td>1.320000e+06</td>
      <td>6.0</td>
      <td>No</td>
      <td>Employed, full-time;Independent contractor, fr...</td>
      <td>6</td>
      <td>4</td>
      <td>Developer, full-stack</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>89179</th>
      <td>89179</td>
      <td>NaN</td>
      <td>Brazil</td>
      <td>BRL\tBrazilian real</td>
      <td>NaN</td>
      <td>HTML/CSS;Java;JavaScript;SQL;TypeScript</td>
      <td>Remote</td>
      <td>2.000000e+05</td>
      <td>NaN</td>
      <td>Yes</td>
      <td>Employed, full-time;Independent contractor, fr...</td>
      <td>20</td>
      <td>5</td>
      <td>Developer, front-end</td>
    </tr>
    <tr>
      <th>89180</th>
      <td>89180</td>
      <td>NaN</td>
      <td>Romania</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Dart;Java;Python;SQL</td>
      <td>Hybrid (some remote, some in-person)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Yes</td>
      <td>Student, full-time;Employed, part-time</td>
      <td>5</td>
      <td>NaN</td>
      <td>Developer, back-end</td>
    </tr>
    <tr>
      <th>89181</th>
      <td>89181</td>
      <td>NaN</td>
      <td>Israel</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Assembly;Bash/Shell (all shells);C;C#;Python;R...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>I prefer not to say</td>
      <td>10</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>89182</th>
      <td>89182</td>
      <td>NaN</td>
      <td>Switzerland</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Bash/Shell (all shells);C#;HTML/CSS;Java;JavaS...</td>
      <td>Hybrid (some remote, some in-person)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>No</td>
      <td>Employed, part-time;Student, part-time</td>
      <td>3</td>
      <td>NaN</td>
      <td>System administrator</td>
    </tr>
    <tr>
      <th>89183</th>
      <td>89183</td>
      <td>NaN</td>
      <td>Iran, Islamic Republic of...</td>
      <td>IRR\tIranian rial</td>
      <td>NaN</td>
      <td>C#;Go;JavaScript;SQL;TypeScript</td>
      <td>Hybrid (some remote, some in-person)</td>
      <td>3.300000e+09</td>
      <td>14.0</td>
      <td>Yes</td>
      <td>Employed, full-time</td>
      <td>17</td>
      <td>12</td>
      <td>Developer, full-stack</td>
    </tr>
  </tbody>
</table>
<p>89184 rows × 14 columns</p>
</div>



**FIX**:  `df['YearsCode'].mean()` TypeError: unsupported operand type(s) for +: 'int' and 'str'


```python
df['YearsCode'].unique()
```




    array([nan, '18', '27', '12', '6', '21', '4', '5', '20', '14', '10', '15',
           '11', '3', '24', '8', '13', 'Less than 1 year', '16', '33', '22',
           '30', '32', '7', '35', '28', '40', '17', '29', '19',
           'More than 50 years', '9', '38', '26', '34', '25', '2', '45', '23',
           '31', '43', '1', '48', '41', '50', '39', '42', '37', '36', '44',
           '46', '49', '47'], dtype=object)




```python
df['YearsCode'].replace(['Less than 1 year', 'More than 50 years'], [1, 51], inplace=True)
df['YearsCode'] = df['YearsCode'].astype(float)

df['YearsCode'].unique()
```




    array([nan, 18., 27., 12.,  6., 21.,  4.,  5., 20., 14., 10., 15., 11.,
            3., 24.,  8., 13.,  1., 16., 33., 22., 30., 32.,  7., 35., 28.,
           40., 17., 29., 19., 51.,  9., 38., 26., 34., 25.,  2., 45., 23.,
           31., 43., 48., 41., 50., 39., 42., 37., 36., 44., 46., 49., 47.])




```python
df['YearsCode'].mean()
```




    13.977926459655745


