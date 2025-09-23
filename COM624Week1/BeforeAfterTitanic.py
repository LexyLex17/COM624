import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df_raw = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
df_cleaned = pd.read_csv('titanic_cleaned.csv')
plt.figure(figsize=(10,6))
plt.imshow(df_raw.isnull(),aspect='auto', interpolation='none')
plt.title("Missing data from Titanic.CSV")
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.xticks(ticks=np.arange(len(df_raw.columns)), labels=df_raw.columns, rotation=45)
plt.colorbar(label='Missing = White, Present = Black')
plt.tight_layout()
plt.savefig("missing_data_before_cleaning_titanic.png")

plt.figure(figsize=(10,6))
plt.imshow(df_cleaned.isnull(),aspect='auto', interpolation='none')
plt.title("Missing data from Titanic.CSV")
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.xticks(ticks=np.arange(len(df_cleaned.columns)), labels=df_cleaned.columns, rotation=45)
plt.colorbar(label='Missing = White, Present = Black')
plt.tight_layout()
plt.savefig("missing_data_after_cleaning_titanic.png")