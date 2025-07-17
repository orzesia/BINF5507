# import all necessary libraries here
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


# 1. Impute Missing Values
def impute_missing_values(data, mean_columns, median_columns, mode_columns):
    """
    Fill missing values in the dataset.
    :param data: pandas DataFrame
    :param strategy: str, imputation method ('mean', 'median', 'mode')
    :return: pandas DataFrame
    """
    for column in mean_columns:
        data[column] = data[column].fillna(data[column].mean())
    for column in median_columns:
        data[column] = data[column].fillna(data[column].median())
    for column in mode_columns:
        data[column] = data[column].fillna(data[column].mode()[0])
    return data

# 2. Remove Duplicates
def remove_duplicates(data):
    """
    Remove duplicate rows from the dataset.
    :param data: pandas DataFrame
    :return: pandas DataFrame
    """
    return data.drop_duplicates()

# 3. Normalize Numerical Data
def normalize_data(data, numeric_columns):
    """Apply normalization to numerical features.
    :param data: pandas DataFrame
    :param method: str, normalization method ('minmax' (default) or 'standard')
    """
    data[numeric_columns] = MinMaxScaler().fit_transform(data[numeric_columns])
    return data

# 4. Remove Redundant Features   - not used
def remove_redundant_features(data, threshold=0.9):
    """Remove redundant or duplicate columns.
    :param data: pandas DataFrame
    :param threshold: float, correlation threshold
    :return: pandas DataFrame
    """
    # TODO: Remove redundant features based on the correlation threshold (HINT: you can use the corr() method)
    pass

# 5. Remove columns with >25% missing values
def remove_columns_missing_data(data, threshold = 0.25):
    """Remove columns with too many missing values.
    :param data: pandas DataFrame
    :param threshold: float, maximum allowed fraction of missing values (default 0.25)
    :return: pandas DataFrame
    """

    max_missing = int((1 - threshold)*len(data))
    return data.dropna(axis=1, thresh=max_missing)

# 6. How many rows have more than 25% of data missing?
def count_missing_data_rows(data, threshold = 0.25):
    """
    Counts rows with missing data above threshold
    :param data: pandas DataFrame
    :param threshold: max fraction of missing data
    :return: fraction of rows with missing data above threshold
    """

    count_missing = data.isnull().mean(axis=1)
    return (sum((count_missing >= threshold))/len(data))

# 5. Remove rows with >25% missing values
def remove_rows_missing_data(data, threshold = 0.25):
    """
    removes rows with missing values above threshold.
    :param data: pandas DataFrame
    :param threshold:max fraction of missing data
    :return: data frame without rows with missing data
    """

    count_missing = data.isnull().mean(axis=1)
    return data[count_missing < threshold]

# 6. Remove outliers (IQR)

def remove_outliers(data, columns):
    """
    Remove rows where the column value is an outlier based on IQR.
    :param data: pandas DataFrame
    :param column: str, column name to check
    :return: pandas DataFrame with outliers removed
    """
    for column in columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return data


# ---------------------------------------------------

def simple_model(input_data, split_data=True, scale_data=False, print_report=False):
    """
    A simple logistic regression model for target classification.
    Parameters:
    input_data (pd.DataFrame): The input data containing features and the target variable 'target' (assume 'target' is the first column).
    split_data (bool): Whether to split the data into training and testing sets. Default is True.
    scale_data (bool): Whether to scale the features using StandardScaler. Default is False.
    print_report (bool): Whether to print the classification report. Default is False.
    Returns:
    None
    The function performs the following steps:
    1. Removes columns with missing data.
    2. Splits the input data into features and target.
    3. Encodes categorical features using one-hot encoding.
    4. Splits the data into training and testing sets (if split_data is True).
    5. Scales the features using StandardScaler (if scale_data is True).
    6. Instantiates and fits a logistic regression model.
    7. Makes predictions on the test set.
    8. Evaluates the model using accuracy score and classification report.
    9. Prints the accuracy and classification report (if print_report is True).
    """

    # if there's any missing data, remove the columns
    input_data.dropna(inplace=True)

    # split the data into features and target
    target = input_data.copy()[input_data.columns[0]]
    features = input_data.copy()[input_data.columns[1:]]

    # if the column is not numeric, encode it (one-hot)
    for col in features.columns:
        if features[col].dtype == 'object':
            features = pd.concat([features, pd.get_dummies(features[col], prefix=col)], axis=1)
            features.drop(col, axis=1, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, stratify=target, random_state=42)

    if scale_data:
        # scale the data
        X_train = normalize_data(X_train)
        X_test = normalize_data(X_test)

    # instantiate and fit the model
    log_reg = LogisticRegression(random_state=42, max_iter=100, solver='liblinear', penalty='l2', C=1.0)
    log_reg.fit(X_train, y_train)

    # make predictions and evaluate the model
    y_pred = log_reg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy}')

    # if specified, print the classification report
    if print_report:
        print('Classification Report:')
        print(report)
        print('Read more about the classification report: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html and https://www.nb-data.com/p/breaking-down-the-classification')

    return None