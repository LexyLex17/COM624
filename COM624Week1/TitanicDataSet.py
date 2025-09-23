import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
df.head()

# Viewing structure and checking for missing values
## Helps to understand which columns have missing data and the overall structure of the dataset
df.info()
df.isnull().sum()

# Drop columns are not useful for analysis
## These columns may  not contribute meaningfully to analysis or prediction
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Filling missing 'age' with median
## Filling missing values helps maintain dataset integrity without removing valuable rows
df['Age'].fillna(df['Embarked'].mode()[0], inplace=True)

# Convert 'Sex' to numeric

df['Sex'] = df['Sex'].map({'male':0, 'female':1})

# Convert 'Embarked' to dummy variables
## Machine learning models require numerical input, so categorical variables must be encoded
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Saving cleaned dataset
## This allows you to reuse the cleaned data in future analysis or modelling tasks
df.to_csv('titanic_cleaned.csv', index=False)