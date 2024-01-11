#linear regression the model performance output is


import pandas as pd
link="C:\\Users\\GARIMA KAPOOR\\Downloads\\jobs_in_data.csv"
df=pd.read_csv(link)
print(df)
print(df.info())
# Print static Dicribe
print(df.describe())
#check the null values , check the duplicates
print(df.isnull().sum())
print(df.drop_duplicates(inplace=True))
print(df.duplicated().sum())
# We will drop the salary  and salary_currency columns from our data and use salary
# in salary_in_usd because it may cause leakage while training and it will affect the model performance
df=df.drop(columns=['salary','salary_currency'])
print(df.head())

import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(df.isnull(), cmap=sns.color_palette(['yellow','red']))
plt.show()
sns.histplot(df.salary_in_usd,kde=True)
plt.savefig("histplot.png")
plt.show()
#top 10 countries which pay the highest salaries
area_salary=df.groupby('employee_residence')['salary_in_usd'].mean().sort_values(ascending=False)
sns.barplot(x=area_salary.nlargest(10).index,y=area_salary.nlargest(10).values)
plt.xticks(rotation=90);
plt.show()

#job category has the highest salary
category_salary=df.groupby('job_category')['salary_in_usd'].mean().sort_values(ascending=False)
category_salary.plot(kind='bar')
plt.savefig("bar.png")
plt.show()

 #relationbetween experience level and salary

sns.scatterplot(x=df.experience_level,y=df.salary_in_usd)
plt.savefig("scatter.png")
plt.show()

#Type of work paid the highest salary
work_type_salary=df.groupby('employment_type')['salary_in_usd'].mean()
work_type_salary.plot(kind='pie',autopct='%1.1f%%')
plt.savefig("pai.png")
plt.show()

df.boxplot()
plt.show()

#relation between numerical columns
df_num=df.select_dtypes(include='number')
df_num.corr()
print(df_num)

#split the data into Train test
from sklearn.model_selection import train_test_split
from sklearn.linear_model  import LinearRegression

X=df.drop(columns='salary_in_usd',axis=1 )
y=df.salary_in_usd.values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=1)
print(X_train.shape)
print(X_test.shape)
#categorical data convert it to numeric
#we will use OrdinalEncoder and pandas get dymmies

from sklearn.preprocessing import OrdinalEncoder
ordinal_inco=OrdinalEncoder()
x_train=ordinal_inco.fit_transform(X_train)
x_test=ordinal_inco.fit_transform(X_test)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.feature_selection import mutual_info_regression
feauture_name=[]
importance_value=[]
columns=ordinal_inco.get_feature_names_out()
imp_feature=mutual_info_regression(x_train,y_train)
for feature,importance in zip(columns,imp_feature):
    feauture_name.append(feature)
    importance_value.append(importance)

    sns.barplot(x=feauture_name, y=importance_value)
    plt.title('Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.savefig("barplot.png")
    plt.show()
