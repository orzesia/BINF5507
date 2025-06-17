### Admin stuff ###

import data_preprocessor as dp
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, precision_recall_curve, average_precision_score
import pandas as pd




### loading data ###
messy_data = pd.read_csv(r'c:\Users\orzes\OneDrive\Desktop\Humber\BINF5507_MLAI\Assignment 2\heart_disease_uci.csv', na_values=['', 'NA', 'Na', 'null', ""])
working_data = messy_data.copy()

### cleaning the data ###
working_data = working_data.drop_duplicates()
working_data = dp.remove_columns_missing_data(working_data)
working_data = dp.remove_rows_missing_data(working_data)
# target columns: "chol" and "num"
working_data = working_data.dropna(subset = "chol")

# for column in ["trestbps","chol","thalch","oldpeak"]:
#     sns.boxplot(x=working_data[column])
#     plt.title(f"Boxplot of {column}")
#     plt.show()

columns = ["trestbps","chol","thalch","oldpeak"]
working_data = dp.remove_outliers(working_data,columns)

working_data = working_data.dropna(subset = "fbs") # just 8
working_data = working_data.dropna(subset = "restecg") # just 1
clean_data = working_data

working_data.to_csv(r"c:\Users\orzes\OneDrive\Desktop\Humber\BINF5507_MLAI\Assignment 2\clean_data.csv", index=False)

# print(working_data[["trestbps","chol","thalch","oldpeak","num"]].corr())
# no correlation found


### split data into training and test ###
# define x and y #
X = clean_data.drop(columns='chol')
y = clean_data['chol']

# Convert categorical columns to numeric using one-hot encoding
X = pd.get_dummies(X, drop_first=True)  # apparently "male" is not scaler. this is what google suggested

# split data into training and testing #
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# standardize features #
scaler = StandardScaler()
scaled_x_train = scaler.fit_transform(X_train)
scaled_x_test = scaler.transform(X_test)

### Train a linear regression model ###

alphas = [0.1, 1, 10]
l1_ratios = [0.1, 0.5, 0.9]
results=[]

for alpha in alphas:
    for l1_ratio in l1_ratios:
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        model.fit(scaled_x_train, y_train)
        y_pred = model.predict(scaled_x_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        results.append((alpha, l1_ratio, rmse, r2))

results_df = pd.DataFrame(results, columns=["alpha", "l1_ratio", "RMSE", "R2"])
results_df.to_csv(r"c:\Users\orzes\OneDrive\Desktop\Humber\BINF5507_MLAI\Assignment 2\results_df.csv", index=False)

# hyperparameter grid
param_grid = {
    'alpha': [0.1, 1.0, 10.0],
    'l1_ratio': [0.1, 0.5, 0.9],
}

grid_search = GridSearchCV(estimator=ElasticNet(random_state=42), param_grid=param_grid)
grid_search.fit(scaled_x_train, y_train)

print('Best alpha:', grid_search.best_estimator_.alpha)
print('Best l1_ratio:', grid_search.best_estimator_.l1_ratio)






### Plot R² heatmap ###
r2_pivot = results_df.pivot(index="alpha", columns="l1_ratio", values="R2")
sns.heatmap(r2_pivot, annot=True)
plt.title("R² Heatmap (ElasticNet)")
plt.xlabel("l1_ratio")
plt.ylabel("alpha")
plt.show()
#
### Plot RMSE heatmap ###
rmse_pivot = results_df.pivot(index="alpha", columns="l1_ratio", values="RMSE")
sns.heatmap(rmse_pivot, annot=True)
plt.title("RMSE Heatmap (ElasticNet)")
plt.xlabel("l1_ratio")
plt.ylabel("alpha")
plt.show()

# ### Print best config ###
# best_r2 = results_df.loc[results_df['R2'].idxmax()]
# print("Best R² config:")
# print(best_r2)
#
# best_rmse = results_df.loc[results_df['RMSE'].idxmin()]
# print("\nBest RMSE config:")
# print(best_rmse)
#
# # To compare the results - minmax normalize and find the highest sum
# # Normalize metrics
# # Add 2 columns to results
# results_df["R2_scaled"] = dp.normalize_data(results_df[["R2"]].copy(), ["R2"])["R2"]
# # results_df.to_csv(r"c:\Users\orzes\OneDrive\Desktop\Humber\BINF5507_MLAI\Assignment 2\results_df.csv", index=False)
#
# results_df["RMSE_scaled"] = dp.normalize_data(results_df[["RMSE"]].copy(), ["RMSE"])["RMSE"]
# results_df["RMSE_scaled_reversed"] = 1- dp.normalize_data(results_df[["RMSE"]].copy(), ["RMSE"])["RMSE"]
#
# results_df["final_score"] = results_df["R2_scaled"] + results_df["RMSE_scaled_reversed"]
# results_df.to_csv(r"c:\Users\orzes\OneDrive\Desktop\Humber\BINF5507_MLAI\Assignment 2\results_df.csv", index=False)





### Classification Setup ###
X = clean_data.drop(columns='num')
y = clean_data['num']
X = pd.get_dummies(X, drop_first=True)

# Binary classification (assumes 'num' is 0 or 1; if not, binarize it)
y = (y > 0).astype(int)  # convert to 0/1 for presence of disease

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

### 1. Logistic Regression (Try different solvers & penalties) ###
solvers = ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]
penalties = ['l1', 'l2']
results = []
best_score = 0
best_log_model = None
log_best_fpr = log_best_tpr = log_best_precision = log_best_recall = None

for solver in solvers:
    for penalty in penalties:
        try:
            model = LogisticRegression(solver=solver, penalty=penalty, max_iter=10000)
            model.fit(X_train_scaled, y_train)
            y_scores = model.predict_proba(X_test_scaled)[:, 1]
            y_pred = model.predict(X_test_scaled)

            # evaluation metrics:
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            fpr, tpr, _ = roc_curve(y_test, y_scores)
            auroc = auc(fpr, tpr) #area under, higher = better
            precision, recall, _ = precision_recall_curve(y_test, y_scores)
            auprc = average_precision_score(y_test, y_scores)

            score = accuracy + f1 + auroc + auprc
            if score > best_score:
                best_score = score
                best_log_model = model
                log_best_fpr = fpr
                log_best_tpr = tpr
                log_best_precision = precision
                log_best_recall = recall

            #print(f"pair: {solver} - {penalty}")
            results.append((solver, penalty, accuracy, f1, auroc, auprc, fpr, tpr, precision, recall))
        except Exception as e:
            #print(f"Skipped: solver={solver}, penalty={penalty} — {e}")
            continue

results_summary = pd.DataFrame(results, columns=["solver", "penalty", "accuracy", "f1", "auroc", "auprc", "_", "_", "_", "_"])
results_summary = results_summary.drop(columns=["_", "_", "_", "_"])
# results_summary.to_csv("results_summary.csv", index=False)

# find the best config:
results_summary["sum"] = results_summary[["accuracy", "f1", "auroc", "auprc"]].sum(axis=1)
results_summary.to_csv("results_summary.csv", index=False)


### 2. k-NN Classification ###
k_values = [1, 5, 10]
best_knn_score = 0
best_knn_model = None
knn_best_fpr = knn_best_tpr = knn_best_precision = knn_best_recall = None
knn_results = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, metric='manhattan')
    knn.fit(X_train_scaled, y_train)
    y_scores_knn = knn.predict_proba(X_test_scaled)[:, 1]
    y_pred_knn = knn.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred_knn)
    f1 = f1_score(y_test, y_pred_knn)
    fpr, tpr, _ = roc_curve(y_test, y_scores_knn)
    auroc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_test, y_scores_knn)
    auprc = average_precision_score(y_test, y_scores_knn)

    score = accuracy + f1 + auroc + auprc
    if score > best_knn_score:
        best_knn_score = score
        best_knn_model = knn
        knn_best_fpr, knn_best_tpr = fpr, tpr
        knn_best_precision, knn_best_recall = precision, recall

    knn_results.append((k, accuracy, f1, auroc, auprc))

knn_results_summary = pd.DataFrame(knn_results, columns=["k", "accuracy", "f1", "auroc", "auprc"])
knn_results_summary["sum"] = knn_results_summary[["accuracy", "f1", "auroc", "auprc"]].sum(axis=1)
knn_results_summary.to_csv("knn_results_summary.csv", index=False)


# Best logistic regression model curves
log_auprc = average_precision_score(y_test, best_log_model.predict_proba(X_test_scaled)[:, 1])
dp.plot_curves(log_best_tpr, log_best_fpr, auc(log_best_fpr, log_best_tpr), log_best_precision, log_best_recall, log_auprc, 'Logistic Regression', minority_class=y.mean())

# Best k-NN model curves
knn_auprc = average_precision_score(y_test, best_knn_model.predict_proba(X_test_scaled)[:, 1])
dp.plot_curves(knn_best_tpr, knn_best_fpr, auc(knn_best_fpr, knn_best_tpr), knn_best_precision, knn_best_recall, knn_auprc, 'k-NN', minority_class=y.mean())