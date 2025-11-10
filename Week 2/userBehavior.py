import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

df = pd.read_csv("user_behavior_dataset.csv")
df.dropna(inplace=True)
df = df[df['Screen_On_Time']>0]

mean_screen = df['Screen_On_Time'].mean()
median_screen = df['Screen_On_Time'].median()
mode_screen = df['Screen_On_Time'].mode()[0]
std_screen = df['Screen_On_Time'].std()
range_screen = df['Screen_On_Time'].max()-df['Screen_On_Time'].min()
print(f"Mean: {mean_screen:.2f}, Median: {median_screen:.2f}, Mode: {mode_screen:.2f}")
print(f"Standard Deviation: {std_screen:.2f}, Range: {range_screen:.2f}")

plt.figure(figsize=(8,5))
sns.histplot(df['Screen_On_Time'], bins=20, kde=True)
plt.title("Distribution of Screen-On time")
plt.xlabel("Hours per day")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x=df['App_Usage_Time'])
plt.title("Box Plot of App Usage Time")
plt.xlabel("Minutes per Day")
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation between variables")
plt.show()

os_group = df.groupby('Operating_System')['Screen_On_Time'].mean()
os_group.plot(kind='bar', color=['skyblue','salmon'])
plt.title("Average Screen-On Time by operating system")
plt.ylabel("Hours per day")
plt.show()

print(df.describe())