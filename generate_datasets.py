import pandas as pd
from sklearn.datasets import load_wine

# 1. WINE
wine_data = load_wine(as_frame=True)
df_wine = wine_data.frame
df_wine.to_csv("datasets/wine.csv", index=False)
print("wine.csv сохранён")

# 2. WHOLESALE (from UCI)
wholesale_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv"
df_wholesale = pd.read_csv(wholesale_url)
df_wholesale.to_csv("datasets/wholesale.csv", index=False)
print("wholesale.csv сохранён")

# 3. MALL CUSTOMERS 
# Manually from kaggle: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python 
# I changed the name to mall.csv