# Read the dataset
import pandas as pd
df = pd.read_csv('/Users/DCL/Downloads/dataset_thyroid_sick (1).csv')

# Data Preprocessing (Drop Feature)
import numpy as np
df = df.replace({"?":np.NAN})
df.isnull().sum()
df=df.drop(["TBG"],axis='columns')

# Data Preprocessing (Label Encoding)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = label_encoder.fit_transform(df[column])
        
# Data Preprocessing (Fill-missing value)
cols = df.columns[df.dtypes.eq('object')]
df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
df= df.interpolate(method = 'median', order = 4)
df.isna().sum()

# Data Preprocessing (Balance dataset using BOO-ST)
x=df.iloc[:,:-1]
y=df.iloc[:,28]
from collections import Counter
print(sorted(Counter(y).items()))
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.20)
from sklearn.ensemble import AdaBoostClassifier
from imblearn.combine import SMOTETomek
def adaboost(x, y):
    model = AdaBoostClassifier()
    model.fit(x,y)
    y_pred = model.predict(x)
    return y_pred
sm = SMOTETomek()
x_sm, y_sm = sm.fit_resample(x, y)
y_smote = adaboost(x_sm, y_sm)
from collections import Counter
print(sorted(Counter(y_smote).items()))

# Feature Selection (Uuivariate- SelectKBest)
from sklearn.feature_selection import SelectKBest, f_classif
k = 14
selector = SelectKBest(score_func=f_classif, k=k)
selector.fit(x_sm, y_sm)
selected_features_indices = selector.get_support(indices=True)
selected_features_names = x_sm.columns[selected_features_indices]
print("Selected Features:", selected_features_names)
feature_scores = selector.scores_[selected_features_indices]
feature_importance = dict(zip(selected_features_names, feature_scores))
print("Importance Scores of Selected Features:")
for feature, score in feature_importance.items():
    print(f"{feature}: {score}")
UVS = x_sm[selected_features_names]

# Feature Selection (Mutual Information Gain)
from sklearn.feature_selection import mutual_info_classif
im2 = mutual_info_classif(x_sm,y_sm)
Im_Feature = pd.Series(im2, x_sm.columns)
Im_Feature.nlargest(14)
top_14_features = Im_Feature.nlargest(14)
selected_features_names_igs = top_14_features.index.tolist()
IGS = x_sm[selected_features_names_igs]

# Parameter Analysis for DSHM Using PFI (For UVS selected features)
import eli5
from eli5.sklearn import PermutationImportance
X_trainu, X_testu, y_trainu, y_testu = train_test_split(UVS, y_sm, test_size=0.2)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_trainu, y_trainu)
# Perform permutation feature importance
perm = PermutationImportance(clf, random_state=1).fit(X_trainu, y_trainu)
eli5.show_weights(perm, feature_names = UVS.columns.tolist())

from sklearn.svm import SVC
clf = SVC()
clf.fit(X_trainu, y_trainu)
# Perform permutation feature importance
perm = PermutationImportance(clf, random_state=1).fit(X_trainu, y_trainu)
eli5.show_weights(perm, feature_names = UVS.columns.tolist())

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
clf.fit(X_trainu, y_trainu)
# Perform permutation feature importance
perm = PermutationImportance(clf, random_state=1).fit(X_trainu, y_trainu)
eli5.show_weights(perm, feature_names = UVS.columns.tolist())

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_trainu, y_trainu)
# Perform permutation feature importance
perm = PermutationImportance(clf, random_state=1).fit(X_trainu, y_trainu)
eli5.show_weights(perm, feature_names = UVS.columns.tolist())

from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier()
clf.fit(X_trainu, y_trainu)
# Perform permutation feature importance
perm = PermutationImportance(clf, random_state=1).fit(X_trainu, y_trainu)
eli5.show_weights(perm, feature_names = UVS.columns.tolist())

from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier()
clf.fit(X_trainu, y_trainu)
# Perform permutation feature importance
perm = PermutationImportance(clf, random_state=1).fit(X_trainu, y_trainu)
eli5.show_weights(perm, feature_names = UVS.columns.tolist())

# Parameter Analysis for DSHM Using PFI (For IGS selected features)
X_traini, X_testi, y_traini, y_testi = train_test_split(IGS, y_sm, test_size=0.2)
clf = DecisionTreeClassifier()
clf.fit(X_traini, y_traini)
perm = PermutationImportance(clf, random_state=1).fit(X_traini, y_traini)
eli5.show_weights(perm, feature_names = IGS.columns.tolist())

clf = SVC()
clf.fit(X_traini, y_traini)
# Perform permutation feature importance
perm = PermutationImportance(clf, random_state=1).fit(X_traini, y_traini)
eli5.show_weights(perm, feature_names = IGS.columns.tolist())

clf = KNeighborsClassifier()
clf.fit(X_traini, y_traini)
perm = PermutationImportance(clf, random_state=1).fit(X_traini, y_traini)
eli5.show_weights(perm, feature_names = IGS.columns.tolist())

clf = RandomForestClassifier()
clf.fit(X_traini, y_traini)
# Perform permutation feature importance
perm = PermutationImportance(clf, random_state=1).fit(X_traini, y_traini)
eli5.show_weights(perm, feature_names = IGS.columns.tolist())

clf = AdaBoostClassifier()
clf.fit(X_traini, y_traini)
# Perform permutation feature importance
perm = PermutationImportance(clf, random_state=1).fit(X_traini, y_traini)
eli5.show_weights(perm, feature_names = IGS.columns.tolist())

clf = GradientBoostingClassifier()
clf.fit(X_traini, y_traini)
perm = PermutationImportance(clf, random_state=1).fit(X_traini, y_traini)
eli5.show_weights(perm, feature_names = IGS.columns.tolist())

# Analysis of ensemble methods
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
base_classifier = DecisionTreeClassifier()
bagging_classifier = BaggingClassifier(base_classifier, n_estimators=10, random_state=42)
bagging_classifier.fit(X_trainu, y_trainu)
y_pred = bagging_classifier.predict(X_testu)
accuracy = accuracy_score(y_testu, y_pred)
print("Bagging Classifier Accuracy (UVS):", accuracy)
bagging_classifier = BaggingClassifier(base_classifier, n_estimators=10, random_state=42)
bagging_classifier.fit(X_traini, y_traini)
y_pred = bagging_classifier.predict(X_testi)
accuracy = accuracy_score(y_testi, y_pred)
print("Bagging Classifier Accuracy (IGS):", accuracy)

boosting = AdaBoostClassifier()
boosting.fit(X_trainu, y_trainu)
y_pred = boosting.predict(X_testu)
accuracy = accuracy_score(y_testu, y_pred)
print("Boosting Classifier Accuracy (UVS):", accuracy)
boosting.fit(X_traini, y_traini)
y_pred = boosting.predict(X_testi)
accuracy = accuracy_score(y_testi, y_pred)
print("Boosting Classifier Accuracy (IGS):", accuracy)

from sklearn.ensemble import VotingClassifier
decision_tree = DecisionTreeClassifier()
voting = VotingClassifier(estimators=[('dt', decision_tree)])
voting.fit(X_trainu, y_trainu)
y_pred = voting.predict(X_testu)
accuracy = accuracy_score(y_testu, y_pred)
print("Voting Classifier Accuracy (UVS):", accuracy)
voting.fit(X_traini, y_traini)
y_pred = voting.predict(X_testi)
accuracy = accuracy_score(y_testi, y_pred)
print("Voting Classifier Accuracy (IGS):", accuracy)

from sklearn.ensemble import StackingClassifier
stacking_classifier = StackingClassifier(estimators=[('dt', decision_tree)], final_estimator=RandomForestClassifier())
stacking_classifier.fit(X_trainu, y_trainu)
y_pred = stacking_classifier.predict(X_testu)
accuracy = accuracy_score(y_testu, y_pred)
print("Stacking Classifier Accuracy (UVS):", accuracy)
stacking_classifier.fit(X_traini, y_traini)
y_pred = stacking_classifier.predict(X_testi)
accuracy = accuracy_score(y_testi, y_pred)
print("Stacking Classifier Accuracy (IGS):", accuracy)


# train-test fold for all preproceesed features
X_traina, X_testa, y_traina, y_testa = train_test_split(x_sm, y_sm, test_size=0.2)

# Evaluate the Baseline ML, Ensemble, and proposed DSHM
from sklearn.metrics import precision_score, recall_score, f1_score
# Decision Tree
clf = DecisionTreeClassifier()
clf.fit(X_traina, y_traina)
y_pred = clf.predict(X_testa)
accuracy = accuracy_score(y_testa, y_pred)
precision = precision_score(y_testa, y_pred)
recall = recall_score(y_testa, y_pred)
f1_score = f1_score(y_testa, y_pred)

print("DT Classifier Accuracy (ALL):", accuracy)
print("DT Classifier Precision (ALL):", precision)
print("DT Classifier Recall (ALL):", recall)
print("DT Classifier F1-score (ALL):", f1_score)

clf.fit(X_trainu, y_trainu)
y_pred = clf.predict(X_testu)
accuracy = accuracy_score(y_testu, y_pred)
precision_uvs = precision_score(y_testu, y_pred)
recall_uvs = recall_score(y_testu, y_pred)
f1_score_uvs = f1_score(y_testu, y_pred)

print("DT Classifier Accuracy (UVS):", accuracy)
print("DT Classifier Precision (UVS):", precision_uvs)
print("DT Classifier Recall (UVS):", recall_uvs)
print("DT Classifier F1-score (UVS):", f1_score_uvs)

clf.fit(X_traini, y_traini)
y_pred = clf.predict(X_testi)
accuracy = accuracy_score(y_testi, y_pred)
precision_igs = precision_score(y_testi, y_pred)
recall_igs = recall_score(y_testi, y_pred)
f1_score_igs = f1_score(y_testi, y_pred)

print("DT Classifier Accuracy (IGS):", accuracy)
print("DT Classifier Precision (IGS):", precision_igs)
print("DT Classifier Recall (IGS):", recall_igs)
print("DT Classifier F1-score (IGS):", f1_score_igs)

# Support Vector Machine
clf = SVC()
clf.fit(X_traina, y_traina)
y_pred = clf.predict(X_testa)
accuracy = accuracy_score(y_testa, y_pred)
precision = precision_score(y_testa, y_pred)
recall = recall_score(y_testa, y_pred)
f1_score = f1_score(y_testa, y_pred)

print("SVM Classifier Accuracy (ALL):", accuracy)
print("SVM Classifier Precision (ALL):", precision)
print("SVM Classifier Recall (ALL):", recall)
print("SVM Classifier F1-score (ALL):", f1_score)

clf.fit(X_trainu, y_trainu)
y_pred = clf.predict(X_testu)
accuracy = accuracy_score(y_testu, y_pred)
precision_uvs = precision_score(y_testu, y_pred)
recall_uvs = recall_score(y_testu, y_pred)
f1_score_uvs = f1_score(y_testu, y_pred)

print("SVM Classifier Accuracy (UVS):", accuracy)
print("SVM Classifier Precision (UVS):", precision_uvs)
print("SVM Classifier Recall (UVS):", recall_uvs)
print("SVM Classifier F1-score (UVS):", f1_score_uvs)

clf.fit(X_traini, y_traini)
y_pred = clf.predict(X_testi)
accuracy = accuracy_score(y_testi, y_pred)
precision_igs = precision_score(y_testi, y_pred)
recall_igs = recall_score(y_testi, y_pred)
f1_score_igs = f1_score(y_testi, y_pred)

print("SVM Classifier Accuracy (IGS):", accuracy)
print("SVM Classifier Precision (IGS):", precision_igs)
print("SVM Classifier Recall (IGS):", recall_igs)
print("SVM Classifier F1-score (IGS):", f1_score_igs)


# K-Nearest Neighbors
clf = KNeighborsClassifier()
clf.fit(X_traina, y_traina)
y_pred = clf.predict(X_testa)
accuracy = accuracy_score(y_testa, y_pred)
precision = precision_score(y_testa, y_pred)
recall = recall_score(y_testa, y_pred)
f1_score = f1_score(y_testa, y_pred)

print("KNN Classifier Accuracy (ALL):", accuracy)
print("KNN Classifier Precision (ALL):", precision)
print("KNN Classifier Recall (ALL):", recall)
print("KNN Classifier F1-score (ALL):", f1_score)

clf.fit(X_trainu, y_trainu)
y_pred = clf.predict(X_testu)
accuracy = accuracy_score(y_testu, y_pred)
precision_uvs = precision_score(y_testu, y_pred)
recall_uvs = recall_score(y_testu, y_pred)
f1_score_uvs = f1_score(y_testu, y_pred)

print("KNN Classifier Accuracy (UVS):", accuracy)
print("KNN Classifier Precision (UVS):", precision_uvs)
print("KNN Classifier Recall (UVS):", recall_uvs)
print("KNN Classifier F1-score (UVS):", f1_score_uvs)

clf.fit(X_traini, y_traini)
y_pred = clf.predict(X_testi)
accuracy = accuracy_score(y_testi, y_pred)
precision_igs = precision_score(y_testi, y_pred)
recall_igs = recall_score(y_testi, y_pred)
f1_score_igs = f1_score(y_testi, y_pred)

print("KNN Classifier Accuracy (IGS):", accuracy)
print("KNN Classifier Precision (IGS):", precision_igs)
print("KNN Classifier Recall (IGS):", recall_igs)
print("KNN Classifier F1-score (IGS):", f1_score_igs)


# Random Forest
clf = RandomForestClassifier()
clf.fit(X_traina, y_traina)
y_pred = clf.predict(X_testa)
accuracy = accuracy_score(y_testa, y_pred)
precision = precision_score(y_testa, y_pred)
recall = recall_score(y_testa, y_pred)
f1_score = f1_score(y_testa, y_pred)

print("RF Classifier Accuracy (ALL):", accuracy)
print("RF Classifier Precision (ALL):", precision)
print("RF Classifier Recall (ALL):", recall)
print("RF Classifier F1-score (ALL):", f1_score)

clf.fit(X_trainu, y_trainu)
y_pred = clf.predict(X_testu)
accuracy = accuracy_score(y_testu, y_pred)
precision_uvs = precision_score(y_testu, y_pred)
recall_uvs = recall_score(y_testu, y_pred)
f1_score_uvs = f1_score(y_testu, y_pred)

print("RF Classifier Accuracy (UVS):", accuracy)
print("RF Classifier Precision (UVS):", precision_uvs)
print("RF Classifier Recall (UVS):", recall_uvs)
print("RF Classifier F1-score (UVS):", f1_score_uvs)

clf.fit(X_traini, y_traini)
y_pred = clf.predict(X_testi)
accuracy = accuracy_score(y_testi, y_pred)
precision_igs = precision_score(y_testi, y_pred)
recall_igs = recall_score(y_testi, y_pred)
f1_score_igs = f1_score(y_testi, y_pred)

print("RF Classifier Accuracy (IGS):", accuracy)
print("RF Classifier Precision (IGS):", precision_igs)
print("RF Classifier Recall (IGS):", recall_igs)
print("RF Classifier F1-score (IGS):", f1_score_igs)


# AdaBoost Classifier
clf = AdaBoostClassifier()
clf.fit(X_traina, y_traina)
y_pred = clf.predict(X_testa)
accuracy = accuracy_score(y_testa, y_pred)
precision = precision_score(y_testa, y_pred)
recall = recall_score(y_testa, y_pred)
f1_score = f1_score(y_testa, y_pred)

print("AB Classifier Accuracy (ALL):", accuracy)
print("AB Classifier Precision (ALL):", precision)
print("AB Classifier Recall (ALL):", recall)
print("AB Classifier F1-score (ALL):", f1_score)

clf.fit(X_trainu, y_trainu)
y_pred = clf.predict(X_testu)
accuracy = accuracy_score(y_testu, y_pred)
precision_uvs = precision_score(y_testu, y_pred)
recall_uvs = recall_score(y_testu, y_pred)
f1_score_uvs = f1_score(y_testu, y_pred)

print("AB Classifier Accuracy (UVS):", accuracy)
print("AB Classifier Precision (UVS):", precision_uvs)
print("AB Classifier Recall (UVS):", recall_uvs)
print("AB Classifier F1-score (UVS):", f1_score_uvs)

clf.fit(X_traini, y_traini)
y_pred = clf.predict(X_testi)
accuracy = accuracy_score(y_testi, y_pred)
precision_igs = precision_score(y_testi, y_pred)
recall_igs = recall_score(y_testi, y_pred)
f1_score_igs = f1_score(y_testi, y_pred)

print("AB Classifier Accuracy (IGS):", accuracy)
print("AB Classifier Precision (IGS):", precision_igs)
print("AB Classifier Recall (IGS):", recall_igs)
print("AB Classifier F1-score (IGS):", f1_score_igs)


# Gradient Boost Classifier
clf = GradientBoostingClassifier()
clf.fit(X_traina, y_traina)
y_pred = clf.predict(X_testa)
accuracy = accuracy_score(y_testa, y_pred)
precision = precision_score(y_testa, y_pred)
recall = recall_score(y_testa, y_pred)
f1_score = f1_score(y_testa, y_pred)

print("GB Classifier Accuracy (ALL):", accuracy)
print("GB Classifier Precision (ALL):", precision)
print("GB Classifier Recall (ALL):", recall)
print("GB Classifier F1-score (ALL):", f1_score)

clf.fit(X_trainu, y_trainu)
y_pred = clf.predict(X_testu)
accuracy = accuracy_score(y_testu, y_pred)
precision_uvs = precision_score(y_testu, y_pred)
recall_uvs = recall_score(y_testu, y_pred)
f1_score_uvs = f1_score(y_testu, y_pred)

print("GB Classifier Accuracy (UVS):", accuracy)
print("GB Classifier Precision (UVS):", precision_uvs)
print("GB Classifier Recall (UVS):", recall_uvs)
print("GB Classifier F1-score (UVS):", f1_score_uvs)

clf.fit(X_traini, y_traini)
y_pred = clf.predict(X_testi)
accuracy = accuracy_score(y_testi, y_pred)
precision_igs = precision_score(y_testi, y_pred)
recall_igs = recall_score(y_testi, y_pred)
f1_score_igs = f1_score(y_testi, y_pred)

print("GB Classifier Accuracy (IGS):", accuracy)
print("GB Classifier Precision (IGS):", precision_igs)
print("GB Classifier Recall (IGS):", recall_igs)
print("GB Classifier F1-score (IGS):", f1_score_igs)


#A-VT
decision_tree = DecisionTreeClassifier()
support_vector = SVC()
k_nearest = KNeighborsClassifier()
random = RandomForestClassifier()
adaboost = AdaBoostClassifier()
gradient = GradientBoostingClassifier()
clf = VotingClassifier(estimators=[('dt', decision_tree), ('svm', support_vector), ('knn', k_nearest), ('rf', random), ('ab', adaboost), ('gb', gradient)])
clf.fit(X_traina, y_traina)
y_pred = clf.predict(X_testa)
accuracy = accuracy_score(y_testa, y_pred)
precision = precision_score(y_testa, y_pred)
recall = recall_score(y_testa, y_pred)
f1_score = f1_score(y_testa, y_pred)

print("A-VT Classifier Accuracy (ALL):", accuracy)
print("A-VT Classifier Precision (ALL):", precision)
print("A-VT Classifier Recall (ALL):", recall)
print("A-VT Classifier F1-score (ALL):", f1_score)

clf.fit(X_trainu, y_trainu)
y_pred = clf.predict(X_testu)
accuracy = accuracy_score(y_testu, y_pred)
precision_uvs = precision_score(y_testu, y_pred)
recall_uvs = recall_score(y_testu, y_pred)
f1_score_uvs = f1_score(y_testu, y_pred)

print("A-VT Classifier Accuracy (UVS):", accuracy)
print("A-VT Classifier Precision (UVS):", precision_uvs)
print("A-VT Classifier Recall (UVS):", recall_uvs)
print("A-VT Classifier F1-score (UVS):", f1_score_uvs)

clf.fit(X_traini, y_traini)
y_pred = clf.predict(X_testi)
accuracy = accuracy_score(y_testi, y_pred)
precision_igs = precision_score(y_testi, y_pred)
recall_igs = recall_score(y_testi, y_pred)
f1_score_igs = f1_score(y_testi, y_pred)

print("A-VT Classifier Accuracy (IGS):", accuracy)
print("A-VT Classifier Precision (IGS):", precision_igs)
print("A-VT Classifier Recall (IGS):", recall_igs)
print("A-VT Classifier F1-score (IGS):", f1_score_igs)



# DSHM
from sklearn.linear_model import LogisticRegression
random = RandomForestClassifier()
adaboost = AdaBoostClassifier()
gradient = GradientBoostingClassifier()
clf = StackingClassifier(estimators=[('rf', random), ('ab', adaboost), ('gb', gradient)], final_estimator=LogisticRegression())
clf.fit(X_traina, y_traina)
y_pred = clf.predict(X_testa)
accuracy = accuracy_score(y_testa, y_pred)
precision = precision_score(y_testa, y_pred)
recall = recall_score(y_testa, y_pred)
f1_score = f1_score(y_testa, y_pred)

print("DSHM Classifier Accuracy (ALL):", accuracy)
print("DSHM Classifier Precision (ALL):", precision)
print("DSHM Classifier Recall (ALL):", recall)
print("DSHM Classifier F1-score (ALL):", f1_score)

clf.fit(X_trainu, y_trainu)
y_pred = clf.predict(X_testu)
accuracy = accuracy_score(y_testu, y_pred)
precision_uvs = precision_score(y_testu, y_pred)
recall_uvs = recall_score(y_testu, y_pred)
f1_score_uvs = f1_score(y_testu, y_pred)

print("DSHM Classifier Accuracy (UVS):", accuracy)
print("DSHM Classifier Precision (UVS):", precision_uvs)
print("DSHM Classifier Recall (UVS):", recall_uvs)
print("DSHM Classifier F1-score (UVS):", f1_score_uvs)

clf.fit(X_traini, y_traini)
y_pred = clf.predict(X_testi)
accuracy = accuracy_score(y_testi, y_pred)
precision_igs = precision_score(y_testi, y_pred)
recall_igs = recall_score(y_testi, y_pred)
f1_score_igs = f1_score(y_testi, y_pred)

print("DSHM Classifier Accuracy (IGS):", accuracy)
print("DSHM Classifier Precision (IGS):", precision_igs)
print("DSHM Classifier Recall (IGS):", recall_igs)
print("DSHM Classifier F1-score (IGS):", f1_score_igs)


# Performing pair-wise Statistical Test using Mann-Whitney U Statistical Test
# DT vs SVM
from scipy.stats import mannwhitneyu
DT = ['dt-accuracy-all', 'dt-accuracy-uvs', 'dt-accuracy-igs']  #The 'dt-accuracy-all', 'dt-accuracy-uvs', 'dt-accuracy-igs' should be replaced by the performing accuracy for ALL, UVS, and IGS features using DT classifier
SVM = ['svm-accuracy-all', 'svm-accuracy-uvs', 'svm-accuracy-igs'] #The 'svm-accuracy-all', 'svm-accuracy-uvs', 'svm-accuracy-igs' should be replaced by the performing accuracy for ALL, UVS, and IGS features using SVM classifier
statistic, p_value = mannwhitneyu(DT, SVM)
# Print the results
print("Mann-Whitney U Statistic:", statistic)
print("P-value:", p_value)
# Compare p-value to the significance level (e.g., 0.05)
if p_value < 0.05:
    print("Reject the null hypothesis. There is a statistically significant difference between the two models.")
else:
    print("Fail to reject the null hypothesis. There is no statistically significant difference between the two models.")



# DT vs KNN
DT = ['dt-accuracy-all', 'dt-accuracy-uvs', 'dt-accuracy-igs']  #The 'dt-accuracy-all', 'dt-accuracy-uvs', 'dt-accuracy-igs' should be replaced by the performing accuracy for ALL, UVS, and IGS features using DT classifier
KNN = ['knn-accuracy-all', 'knn-accuracy-uvs', 'knn-accuracy-igs'] #The 'knn-accuracy-all', 'knn-accuracy-uvs', 'knn-accuracy-igs' should be replaced by the performing accuracy for ALL, UVS, and IGS features using KNN classifier
statistic, p_value = mannwhitneyu(DT, KNN)
# Print the results
print("Mann-Whitney U Statistic:", statistic)
print("P-value:", p_value)
# Compare p-value to the significance level (e.g., 0.05)
if p_value < 0.05:
    print("Reject the null hypothesis. There is a statistically significant difference between the two models.")
else:
    print("Fail to reject the null hypothesis. There is no statistically significant difference between the two models.")
    

# DT vs RF
DT = ['dt-accuracy-all', 'dt-accuracy-uvs', 'dt-accuracy-igs']  #The 'dt-accuracy-all', 'dt-accuracy-uvs', 'dt-accuracy-igs' should be replaced by the performing accuracy for ALL, UVS, and IGS features using DT classifier
RF = ['rf-accuracy-all', 'rf-accuracy-uvs', 'rf-accuracy-igs'] #The 'rf-accuracy-all', 'rf-accuracy-uvs', 'rf-accuracy-igs' should be replaced by the performing accuracy for ALL, UVS, and IGS features using RF classifier
statistic, p_value = mannwhitneyu(DT, RF)
# Print the results
print("Mann-Whitney U Statistic:", statistic)
print("P-value:", p_value)
# Compare p-value to the significance level (e.g., 0.05)
if p_value < 0.05:
    print("Reject the null hypothesis. There is a statistically significant difference between the two models.")
else:
    print("Fail to reject the null hypothesis. There is no statistically significant difference between the two models.")

# Please consider all posible pairs of employed models/classifiers using the same way
# Please consider all posible pairs of employed models/classifiers using the same way
# Please consider all posible pairs of employed models/classifiers using the same way
# ....
# ....
# ....
# .... 
# Please consider all posible pairs of employed models/classifiers using the same way
# Please consider all posible pairs of employed models/classifiers using the same way
# Please consider all posible pairs of employed models/classifiers using the same way

 
# Performed the LIME XAI to make the prediction of highest performing DSHM model transparent 
import lime
import lime.lime_tabular
explainer = lime.lime_tabular.LimeTabularExplainer(X_trainu.values,
                                                   mode='classification',
                                                   training_labels=y_trainu,
                                                   feature_names=X_trainu.columns,
                                                   verbose=True)
predict_fn = lambda x: clf.predict_proba(x).astype(float)
# Select a sample instance to explain
sample_idx = 4561 # Randomly selected
# Explain the prediction for the selected instance
exp = explainer.explain_instance(X_testu.values[sample_idx],
                                  predict_fn,
                                  num_features=len(UVS))

# Plot the explanation
exp.show_in_notebook(show_table=True, show_all=False)


# Performed the SHAP XAI to display the high contributory features
import shap
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_testu)
shap.summary_plot(shap_values[1], X_testu)