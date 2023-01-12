from scipy.stats import skew
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


#Fetching datas

URL = 'https://www.worldometers.info/world-population/population-by-country/'
tables = pd.read_html(URL, match="Country")
npp = np.array(tables)
l = np.reshape(npp, (235, 12))
df = pd.DataFrame(l, columns=('#', 'Country', 'Population (2020)', 'Yearly Change', 'Net Change', 'Density (P/Km²)',
                  'Land Area (Km²)', 'Migrants (net)', 'Fert. Rate', 'Med. Age', 'Urban Pop %', 'World Share'))
df.index.name = 'index'
df.drop(labels=['#', 'Migrants (net)'], axis=1, inplace=True)




#Editing features

print('Population:', df['Population (2020)'].mean(axis=0))
df['Yearly Change'] = df['Yearly Change'].str.replace('%', '')
df['Yearly Change'] = df['Yearly Change'].str.replace(" ", "")
df['Yearly Change'] = pd.to_numeric(
    df['Yearly Change'], downcast="float")
print('Yearly Change:',df['Yearly Change'].mean(axis=0))
print('Net Change:',df['Net Change'].mean(axis=0))
print('Density:', df['Density (P/Km²)'].mean(axis=0))
print('Land Area:', df['Land Area (Km²)'].mean(axis=0))
df['Fert. Rate'] = df['Fert. Rate'].str.replace('N.A.', '', regex=True)
df['Fert. Rate'] = pd.to_numeric(
    df['Fert. Rate'], downcast="float")
print('Fert. Rate:', df['Fert. Rate'].mean(axis=0))
df['Med. Age'] = df['Med. Age'].str.replace('N.A.', '', regex=True)
df['Med. Age'] = pd.to_numeric(
    df['Med. Age'], downcast="float")
print('Med. Age:',df['Med. Age'].mean(axis=0))
df['Urban Pop %'] = df['Urban Pop %'].str.replace('%', '')
df['Urban Pop %'] = df['Urban Pop %'].str.replace(' ', '')
df['Urban Pop %'] = df['Urban Pop %'].str.replace('N.A.', '', regex=True)
df['Urban Pop %'] = pd.to_numeric(
    df['Urban Pop %'], downcast="float")
print('Urban Pop:', df['Urban Pop %'].mean(axis=0))
df['World Share'] = df['World Share'].str.replace('%', '')
df['World Share'] = df['World Share'].str.replace(' ', '')
df['World Share'] = df['World Share'].str.replace("N.A.", '', regex=True)
df['World Share'] = pd.to_numeric(
    df['World Share'], downcast="float")
print('World Share:', df['World Share'].mean(axis=0))



#Creating New Features

status = []
for st in df['Density (P/Km²)']:
         if st > int(df['Density (P/Km²)'].mean(axis=0)):
             for s in df['Yearly Change']:
               if s > df['Yearly Change'].mean(axis=0):
                 status.append('Fast Growing')
                 break
         else:
            status.append('Slow Growing')


df['Population Status'] = status

status2 = []
for st2 in df['Urban Pop %']:
         if st2 > int(df['Urban Pop %'].mean(axis=0)):
             for s2 in df['Fert. Rate']:
               if s2 < df['Fert. Rate'].mean(axis=0):
                 status2.append('Developed')
                 break
         else:
            status2.append('Developing')


df['Development Status'] = status2



#Null values

print(df.isnull().sum())
df.fillna(df.mean(numeric_only=True).round(1), inplace=True)
print(df.isna().sum())
print(df['Development Status'].describe())
print(str(df.isnull().values.sum()))



#Most important features

df3=df.copy()
print("Find most important features relative to target")
corr = df.corr()
print(corr)
corr2 = df.corr()[['Yearly Change']]
print(corr2.sort_values)



#Categorical and numerical features

categorical_features = df.select_dtypes(include=['object']).columns
print('categorical_features: ',categorical_features)
numerical_features = df.select_dtypes(exclude = ["object"]).columns
print('numerical_features: ',numerical_features)
train_num = df[numerical_features]
train_cat = df[categorical_features]


#Skewness

skewness = train_num.apply(lambda x: skew(x))
print('skewness: ',skewness.sort_values(ascending=False))


#Preparing for modelling 

y = pd.DataFrame(df['Yearly Change'], columns=['Yearly Change'])
y.index.name = 'index'
df.drop(labels=['Yearly Change'], axis=1, inplace=True)
X = pd.DataFrame(df, columns=('Country', 'Population (2020)', 'Net Change', 'Density (P/Km²)',
                             'Land Area (Km²)', 'Fert. Rate', 'Med. Age', 'Urban Pop %', 'World Share', 'Population Status', 'Development Status'))
X.index.name = 'index'


X = X.apply(pd.to_numeric, errors='coerce')
Y = y.apply(pd.to_numeric, errors='coerce')

X.fillna(0, inplace=True)
y.fillna(0, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0, test_size=0.20)
scaler = StandardScaler()
scaler.fit(X_train)
features = X_train.columns

X_train.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)
X_train[features] = scaler.transform(X_train).astype(float)
X_test[features] = scaler.transform(X_test).astype(float)

print(X_train[features])
print('X Train shape:', X_train.shape)
print('X Test shape:', X_test.shape)
print('Y Train shape:', y_train.shape)
print('Y Test shape:', y_test.shape)

