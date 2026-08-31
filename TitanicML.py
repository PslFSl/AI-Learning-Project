import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('train.csv')
print(df.info())
print(df.head())
print(df.describe())
#Examine data set

print(df.isnull().sum())
#Identifying empty data

df["Age"] = df["Age"].fillna(df["Age"].median()) #Fill in the age using the median (median is preferred over mean, as it is more resistant to outliers)
df = df.drop(columns=["Cabin"]) #There are too many missing in the cabin, skip the column
df = df.dropna(subset=["Embarked"]) #There are a few missing lines in Embarked, skip those

df["Sex"] = df["Sex"].map({"male": 1, "female": 0})
#Converse categorical data to numerical data "encoding"


print(df["Survived"].value_counts())
print(df.groupby("Sex")["Survived"].mean())
print(df.groupby("Pclass")["Survived"].mean())
print(df.groupby(["Sex","Pclass"])["Survived"].mean())
#Exploratory Data Analysis

df["Age"].hist(bins=30)
plt.title("Histogram of Age")
plt.show()
#Histogram of Age

df.groupby("Pclass")["Survived"].mean().plot(kind="bar")
plt.title("Survival Rate by Ticket Class")
plt.show()
#Survival Rate by Ticket Class

plt.scatter(df["Age"], df["Fare"])
plt.xlabel("Age")
plt.ylabel("Ticket Fare")
plt.show()
#Ticket fare by passenger age

x=df[["Pclass","Sex","Age","Fare"]].values
y=df["Survived"].values
print(x.shape,y.shape)
#The pure NumPy array is now ready for ML