###  Admin stuff

# Import necessary modules
import data_preprocessor as dp
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
messy_data = pd.read_csv(r'c:\Users\orzes\OneDrive\Desktop\Humber\BINF5507_MLAI\assignments\messy_data.csv', na_values=['', 'NA', 'Na', 'null', ""])
working_data = messy_data.copy()

# # look into the messy data
# print(messy_data.head())
# print(messy_data.info())
# print(messy_data.describe())
# print(working_data.isnull().sum())

### Preprocessing

# 1. remove duplicates

working_data = dp.remove_duplicates(working_data)
working_data.to_csv(r"c:\Users\orzes\OneDrive\Desktop\Humber\BINF5507_MLAI\assignments\cleaning_step1.csv", index=False)


# 2. remove columns with missing data

working_data = dp.remove_columns_missing_data(working_data)
working_data.to_csv(r"c:\Users\orzes\OneDrive\Desktop\Humber\BINF5507_MLAI\assignments\cleaning_step2.csv", index=False)

# print(working_data.isnull().sum())

# 3.1 fraction of rows that have 25%+ of missing data (6.18%)

fraction = dp.count_missing_data_rows(working_data)
print(f"Fraction of rows with >=25% missing data fraction: {round(fraction,2)}")

# 3.2 remove rows with 25+% missing values
working_data = dp.remove_rows_missing_data(working_data)
working_data.to_csv(r"c:\Users\orzes\OneDrive\Desktop\Humber\BINF5507_MLAI\assignments\cleaning_step3.csv", index=False)

# 4.1 exploration into redundant features
print(working_data[["o","p","q"]].corr())
print(working_data[["r","u","y"]].corr())
# leaving r and p
x=working_data[["b","c","f","h","k","l","n","p","r","t","v","w"]].corr()
x.to_csv(r"c:\Users\orzes\OneDrive\Desktop\Humber\BINF5507_MLAI\assignments\x.csv", index=False)

# 4.2 deleting redundant features
working_data = working_data.drop(columns =["o","q","u","y","c","n"])
working_data.to_csv(r"c:\Users\orzes\OneDrive\Desktop\Humber\BINF5507_MLAI\assignments\cleaning_step4.csv", index=False)

# 5.1 fraction of rows with missing target
fraction = working_data["target"].isnull().mean()
print(f"Fraction of rows with missing target data: {round(fraction,2)}")

# 5.2 delete rows with missing target
working_data = working_data.dropna(subset = "target")
working_data.to_csv(r"c:\Users\orzes\OneDrive\Desktop\Humber\BINF5507_MLAI\assignments\cleaning_step5.csv", index=False)

# visualization

for column in ["b","f","k","l","p","r","v","w","h","t","target","a","i"]:
    sns.boxplot(x=working_data[column])
    plt.title(f"Boxplot of {column}")
    plt.show()

# remove outliers (IQR)
columns = ["b","f","k","p","l","v","w"]
working_data = dp.remove_outliers(working_data,columns)
working_data.to_csv(r"c:\Users\orzes\OneDrive\Desktop\Humber\BINF5507_MLAI\assignments\cleaning_step_outliers.csv", index=False)


# 6. impute missing values
mean_columns = ["b","f","k","l","p","r","v","w"]
median_columns = ["h","t"]
mode_columns = ["target","a","i"]
working_data = dp.impute_missing_values(working_data, mean_columns, median_columns, mode_columns)
working_data.to_csv(r"c:\Users\orzes\OneDrive\Desktop\Humber\BINF5507_MLAI\assignments\cleaning_step6.csv", index=False)

# 7. normalize the data
numeric_columns = mean_columns+median_columns
clean_data = dp.normalize_data(working_data,numeric_columns)
clean_data.to_csv(r"c:\Users\orzes\OneDrive\Desktop\Humber\BINF5507_MLAI\assignments\clean_data.csv", index=False)


### Train and evaluate the model
model_results = dp.simple_model(messy_data)
print(model_results)
model_results = dp.simple_model(clean_data)
print(model_results)
