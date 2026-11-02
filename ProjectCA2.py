# ==========================================================
# Predictive Analytics Project
# Dataset : Global Data on Sustainable Energy
# ==========================================================

import os
import glob
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score,
                             f1_score, log_loss, roc_auc_score, roc_curve,
                             confusion_matrix, ConfusionMatrixDisplay)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ONE size for every figure in the project (width, height in inches).
# 9 x 5 inches fits on every laptop screen, so we never write figsize again.
plt.rcParams["figure.figsize"] = (9, 5)
plt.rcParams["figure.dpi"] = 100

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


# ==========================================================
# UNIT I : LOAD THE DATA
# ==========================================================

df = pd.read_csv(r"C:/Users/chand/Downloads/global-data-on-sustainable-energy (1).csv")
# the original column names are very long, so give short names
df.columns = ["country", "year", "access_elec", "clean_fuel", "renew_cap_pc",
              "fin_flows", "renew_share", "elec_fossil", "elec_nuclear",
              "elec_renew", "low_carbon", "energy_pc", "energy_intensity",
              "co2", "renew_primary", "gdp_growth", "gdp_pc",
              "density", "land_area", "lat", "lon"]

# the density column is text because of commas (example 1,234), make it a number
df["density"] = df["density"].astype(str).str.replace(",", "")
df["density"] = pd.to_numeric(df["density"], errors="coerce")

print("Rows and columns:", df.shape)
print(df.head())
print(df.describe())
print("Number of countries:", df["country"].nunique())
print("Years:", df["year"].min(), "to", df["year"].max())


# ==========================================================
# UNIT I : DATA PREPROCESSING
# ==========================================================

# ---- 1. missing values ----
missing_percent = df.isnull().sum() / len(df) * 100
missing_percent = missing_percent[missing_percent > 0].sort_values(ascending=False)
print("\nMissing values in percent:")
print(missing_percent)

# ---- 2. drop the column which has more than 60% missing ----
data = df.copy()
for col in missing_percent.index:
    if missing_percent[col] > 60:
        print("Dropping column:", col)
        data = data.drop(columns=col)

# ---- 3. fill the remaining missing values with the median ----
number_columns = data.select_dtypes(include=np.number).columns
for col in number_columns:
    data[col] = data[col].fillna(data[col].median())
print("Missing values left:", data.isnull().sum().sum())

# ---- 4. outliers: look at boxplots ----
four_cols = ["co2", "energy_pc", "gdp_pc", "elec_fossil"]
for i in range(4):
    plt.subplot(1, 4, i + 1)
    sns.boxplot(y=data[four_cols[i]], color="skyblue")
    plt.title(four_cols[i])
plt.suptitle("Boxplots - many outliers are present")
plt.tight_layout()
plt.show()

# ---- 5. these columns are very skewed, log reduces the effect of outliers ----
data["log_co2"] = np.log1p(data["co2"])
data["log_energy_pc"] = np.log1p(data["energy_pc"])
data["log_gdp_pc"] = np.log1p(data["gdp_pc"])
data["log_fossil"] = np.log1p(data["elec_fossil"])
data["log_renew"] = np.log1p(data["elec_renew"])


# ==========================================================
# EDA : VISUALISATIONS
# ==========================================================

# ---- 1. histograms ----
hist_cols = ["access_elec", "clean_fuel", "renew_share",
             "low_carbon", "energy_pc", "gdp_pc"]
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.hist(data[hist_cols[i]], bins=25, color="teal")
    plt.title(hist_cols[i])
plt.suptitle("Histograms")
plt.tight_layout()
plt.show()

# ---- 2. average trend over the years ----
yearly = data.groupby("year")[["access_elec", "clean_fuel", "renew_share", "low_carbon"]].mean()
yearly.plot(marker="o")
plt.title("Average of all countries by year")
plt.ylabel("Percentage")
plt.tight_layout()
plt.show()

# ---- 3. electricity by source ----
source = data.groupby("year")[["elec_fossil", "elec_nuclear", "elec_renew"]].sum()
source.plot(kind="area", alpha=0.8)
plt.title("Electricity generation by source (TWh)")
plt.tight_layout()
plt.show()

# ---- 4. top 10 CO2 emitting countries in the latest year ----
latest = data[data["year"] == data["year"].max()]

top_co2 = latest.sort_values("co2", ascending=False).head(10)
plt.barh(top_co2["country"], top_co2["co2"], color="firebrick")
plt.gca().invert_yaxis()
plt.title("Top 10 CO2 emitting countries")
plt.xlabel("CO2 emissions (kt)")
plt.tight_layout()
plt.show()

# ---- 5. correlation heatmap ----
corr_cols = ["access_elec", "clean_fuel", "renew_cap_pc", "renew_share",
             "elec_fossil", "elec_nuclear", "elec_renew", "low_carbon",
             "energy_pc", "energy_intensity", "co2", "renew_primary",
             "gdp_growth", "gdp_pc"]
sns.heatmap(data[corr_cols].corr(), annot=True, fmt=".1f", cmap="coolwarm",
            annot_kws={"size": 7})
plt.title("Correlation matrix")
plt.tight_layout()
plt.show()

print("\nCorrelation of every column with CO2:")
print(data[corr_cols].corr()["co2"].sort_values(ascending=False))

# ==========================================================
# UNIT II : REGRESSION
# ==========================================================

# this function prints the 4 error measures for a regression model
def show_scores(name, y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(name, "| MAE =", round(mae, 3), "| MSE =", round(mse, 3),
          "| RMSE =", round(rmse, 3), "| R2 =", round(r2, 3))


# ---- 1. Simple Linear Regression (one input) ----
# input x = log GDP per capita, output y = log energy per capita
X = data[["log_gdp_pc"]]
y = data["log_energy_pc"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

simple_model = LinearRegression()
simple_model.fit(X_train, y_train)
print("\n--- Simple Linear Regression ---")
print("Slope:", simple_model.coef_[0])
print("Intercept:", simple_model.intercept_)
show_scores("Simple LR", y_test, simple_model.predict(X_test))

x_sorted = X_test.sort_values("log_gdp_pc")
plt.scatter(X_test, y_test, s=10, alpha=0.4, label="actual")
plt.plot(x_sorted, simple_model.predict(x_sorted), color="red", label="regression line")
plt.xlabel("log GDP per capita")
plt.ylabel("log energy per capita")
plt.title("Simple Linear Regression")
plt.legend()
plt.tight_layout()
plt.show()

# ---- 2. Ordinary Least Squares (OLS) and correlation ----
ols = stats.linregress(X_train["log_gdp_pc"], y_train)
print("\n--- OLS result ---")
print("Slope        :", round(ols.slope, 4))
print("Intercept    :", round(ols.intercept, 4))
print("Correlation r:", round(ols.rvalue, 4))
print("R-squared    :", round(ols.rvalue ** 2, 4))
print("p-value      :", ols.pvalue)

# ---- 3. Multiple Linear Regression (many inputs) ----
# predict log CO2 using many columns
features = ["log_gdp_pc", "log_energy_pc", "log_fossil", "log_renew",
            "access_elec", "renew_share", "energy_intensity"]
X = data[features]
y = data["log_co2"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scaling puts all columns on the same scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

multiple_model = LinearRegression()
multiple_model.fit(X_train_s, y_train)
print("\n--- Multiple Linear Regression ---")
show_scores("Multiple LR", y_test, multiple_model.predict(X_test_s))

y_pred = multiple_model.predict(X_test_s)
plt.scatter(y_test, y_pred, s=10, alpha=0.4)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual log CO2")
plt.ylabel("Predicted log CO2")
plt.title("Actual vs Predicted (Multiple LR)")
plt.tight_layout()
plt.show()

# ---- 4. Polynomial Regression (curve instead of line) ----
X = data[["log_gdp_pc"]]
y = data["log_energy_pc"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n--- Polynomial Regression ---")
plt.scatter(X_test, y_test, s=10, alpha=0.3, color="grey")
x_line = np.linspace(X["log_gdp_pc"].min(), X["log_gdp_pc"].max(), 200).reshape(-1, 1)
x_line = pd.DataFrame(x_line, columns=["log_gdp_pc"])
colors = ["red", "blue", "green", "orange"]

for degree in [1, 2, 3, 4]:
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)

    show_scores("Degree " + str(degree), y_test, poly_model.predict(X_test_poly))
    plt.plot(x_line, poly_model.predict(poly.transform(x_line)),
             color=colors[degree - 1], label="degree " + str(degree))

plt.xlabel("log GDP per capita")
plt.ylabel("log energy per capita")
plt.title("Polynomial Regression")
plt.legend()
plt.tight_layout()
plt.show()


# ==========================================================
# UNIT II and III : LOGISTIC REGRESSION + CLASSIFICATION
# ==========================================================

# Target (y): is the low carbon electricity share HIGH (1) or LOW (0)?
# We use the median value as the dividing line.
median_value = data["low_carbon"].median()
data["high_low_carbon"] = (data["low_carbon"] >= median_value).astype(int)
print("\nNumber of rows in each class:")
print(data["high_low_carbon"].value_counts())

# the low_carbon column itself is NOT used as input, otherwise the answer leaks
cls_features = ["access_elec", "clean_fuel", "renew_share", "log_gdp_pc",
                "log_energy_pc", "energy_intensity", "log_co2", "year"]
X = data[cls_features]
y = data["high_low_carbon"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# scaling is important for SVM and KNN
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---- train the models ----
logistic = LogisticRegression(max_iter=1000)
logistic.fit(X_train, y_train)

tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

svm = SVC(kernel="rbf", probability=True, random_state=42)
svm.fit(X_train, y_train)

naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)

# ---- KNN: try k = 1 to 20 and pick the best k ----
k_list = list(range(1, 21))
k_accuracy = []
for k in k_list:
    knn_try = KNeighborsClassifier(n_neighbors=k)
    knn_try.fit(X_train, y_train)
    k_accuracy.append(accuracy_score(y_test, knn_try.predict(X_test)))

best_k = k_list[int(np.argmax(k_accuracy))]
print("\nBest k for KNN:", best_k)

plt.plot(k_list, k_accuracy, marker="o")
plt.xticks(k_list)
plt.xlabel("k (number of neighbours)")
plt.ylabel("Accuracy")
plt.title("KNN - accuracy for different k")
plt.tight_layout()
plt.show()

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)

# ---- logistic regression odds ratio ----
odds = pd.Series(np.exp(logistic.coef_[0]), index=cls_features)
print("\nOdds ratio (above 1 = increases chance of HIGH low-carbon):")
print(odds.sort_values())

# ---- evaluate all models ----
models = {"Logistic": logistic, "Decision Tree": tree, "SVM": svm,
          "Naive Bayes": naive_bayes, "KNN": knn}

rows = []
plot_number = 1
for name in models:
    model = models[name]
    y_pred = model.predict(X_test)                # predicted class 0 or 1
    y_prob = model.predict_proba(X_test)[:, 1]    # probability of class 1

    rows.append([name,
                 accuracy_score(y_test, y_pred),
                 precision_score(y_test, y_pred),
                 recall_score(y_test, y_pred),
                 f1_score(y_test, y_pred),
                 log_loss(y_test, y_prob),
                 roc_auc_score(y_test, y_prob)])

    # confusion matrix of this model (2 rows x 3 columns of small plots)
    plt.subplot(2, 3, plot_number)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Low", "High"], yticklabels=["Low", "High"])
    plt.title(name)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plot_number = plot_number + 1

plt.suptitle("Confusion matrices")
plt.tight_layout()
plt.show()

result = pd.DataFrame(rows, columns=["Model", "Accuracy", "Precision",
                                      "Recall", "F1", "LogLoss", "AUC"])
print("\nModel comparison:")
print(result.round(3).to_string(index=False))

# ---- ROC curve of all models in one graph ----
for name in models:
    y_prob = models[name].predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=name + " (AUC = " + str(round(auc, 2)) + ")")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.show()

# ---- decision tree picture (these are the "rules") ----
plot_tree(tree, feature_names=cls_features, class_names=["Low", "High"],
          filled=True, fontsize=6)
plt.title("Decision Tree (depth 3)")
plt.tight_layout()
plt.show()

# ==========================================================
# UNIT IV : ASSOCIATION RULES (MARKET BASKET ANALYSIS)
# ==========================================================
# Idea: every row (one country in one year) is a "basket".
# The "items" in the basket are things like  co2_Low, gdp_pc_High ...
# We find which items come together often.
#
#   support    = how often the items appear together (fraction of rows)
#   confidence = if A is present, how often B is also present
#   lift       = confidence / how common B is. Lift > 1 means a real link.

# step 1 - divide every column into Low / Med / High (3 equal groups)
basket_cols = ["access_elec", "clean_fuel", "renew_share", "low_carbon",
               "energy_pc", "gdp_pc", "co2", "energy_intensity",
               "elec_renew", "elec_fossil"]

basket = pd.DataFrame()
for col in basket_cols:
    groups = pd.qcut(data[col].rank(method="first"), 3, labels=["Low", "Med", "High"])
    one_hot = pd.get_dummies(groups, prefix=col).astype(bool)   # True / False columns
    basket = pd.concat([basket, one_hot], axis=1)

min_support = 0.15
min_confidence = 0.7

# helper: support of a group of items
def get_support(items):
    return basket[list(items)].all(axis=1).mean()

# step 2 - find frequent single items
frequent_1 = []
for item in basket.columns:
    if get_support([item]) >= min_support:
        frequent_1.append(item)

# step 3 - frequent pairs of items
frequent_2 = []
for pair in combinations(frequent_1, 2):
    if get_support(pair) >= min_support:
        frequent_2.append(pair)

# step 4 - frequent triples (only if all its pairs are frequent)
frequent_3 = []
for triple in combinations(frequent_1, 3):
    all_pairs_ok = all(p in frequent_2 for p in combinations(triple, 2))
    if all_pairs_ok and get_support(triple) >= min_support:
        frequent_3.append(triple)

print("\nFrequent item groups -> single:", len(frequent_1),
      "| pairs:", len(frequent_2), "| triples:", len(frequent_3))

# step 5 - make rules  "A -> B"  from pairs and triples
rule_rows = []
for group in frequent_2 + frequent_3:
    group_support = get_support(group)
    for size in range(1, len(group)):
        for A in combinations(group, size):
            B = tuple(item for item in group if item not in A)
            confidence = group_support / get_support(A)
            lift = confidence / get_support(B)
            if confidence >= min_confidence and lift > 1:
                rule_rows.append([", ".join(A), ", ".join(B),
                                  group_support, confidence, lift])

rules = pd.DataFrame(rule_rows, columns=["If (A)", "Then (B)", "support", "confidence", "lift"])
rules = rules.sort_values("lift", ascending=False)

print("Total rules found:", len(rules))
print("\nTop 15 rules:")
print(rules.head(15).round(2).to_string(index=False))

print("\nProject finished.")
