import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, ConfusionMatrixDisplay, confusion_matrix, roc_curve
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('week4_data2.csv', skiprows=1, names=['x1', 'x2', 'y'])

plt.figure(figsize=[8, 6])
plt.scatter(
    y=df[df['y'] == 1]['x2'],
    x=df[df['y'] == 1]['x1'],
    color='blue',
    marker='+',
    label='y = +1'
)
plt.scatter(
    x=df[df['y'] == -1]['x1'],
    y=df[df['y'] == -1]['x2'],
    color='orange',
    marker='o',
    label='y = -1'
)

plt.title('Dataset 1 with two features and binary label')
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.rcParams.update({
    'font.size': 13,
    'axes.labelweight': 'bold',
    'axes.edgecolor': '#333333',
})

plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1), fontsize='large')
plt.show()

X = df[['x1', 'x2']].to_numpy()
y = df['y'].to_numpy()

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

c_values = [0.01, 0.1, 1, 2, 5, 10, 20]
degree_values = [1, 2, 3, 4, 5, 6, 7, 8]

scores_by_c_degree = {}
for c in c_values:
    lr_model = LogisticRegression(penalty='l2', C=c)
    for d in degree_values:
        pf = PolynomialFeatures(degree=d, include_bias=False)
        x_train_pl = pf.fit_transform(x_train)
        scores = cross_val_score(lr_model, x_train_pl, y_train, cv=5, scoring='roc_auc')
        scores_by_c_degree[(c, d)] = scores

degree_scores = [v for (c, _), v in scores_by_c_degree.items() if c == 1]
c_scores = [v for (_, d), v in scores_by_c_degree.items() if d == 2]

means = [scores.mean() for scores in c_scores]
stds = [scores.std() for scores in c_scores]

plt.figure(figsize=(8, 5))
main_color = '#1f77b4'
error_color = '#999999'
plt.errorbar(
    c_values, means, yerr=stds,
    fmt='o-',
    capsize=5,
    elinewidth=1.5, capthick=1.5,
    color=main_color, ecolor=error_color,
    label='Mean AUC ± 1 SD'
)

plt.xlabel('C', fontsize=12)
plt.ylabel('Mean ROC_AUC', fontsize=12)
plt.title('Logistic Regression Cross-Validation for C', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.legend(loc='best')
plt.show()

means = [scores.mean() for scores in degree_scores]
stds = [scores.std() for scores in degree_scores]

plt.figure(figsize=(8, 5))
main_color = '#1f77b4'
error_color = '#999999'
plt.errorbar(
    degree_values, means, yerr=stds,
    fmt='o-',
    capsize=5,
    elinewidth=1.5, capthick=1.5,
    color=main_color, ecolor=error_color,
    label='Mean AUC ± 1 SD'
)

plt.xlabel('Degree', fontsize=12)
plt.ylabel('Mean ROC_AUC Score', fontsize=12)
plt.title('Logistic Regression Cross-Validation for degree', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.legend(loc='best')
plt.show()

mean_by_c_degree = {key: scores.mean() for key, scores in scores_by_c_degree.items()}
grid = np.array(list(mean_by_c_degree.values())).reshape(len(c_values), len(degree_values))

plt.figure(figsize=(8, 6))
sns.heatmap(grid, annot=True, fmt=".3f", cmap="viridis", cbar=True, linewidths=0.5, linecolor='gray')

plt.xlabel("Polynomial Degree", fontsize=12)
plt.ylabel("C", fontsize=12)
plt.title("Mean ROC AUC for $C$ vs Polynomial Degree $d$", fontsize=14)
plt.xticks(ticks=np.arange(len(degree_values)) + 0.5, labels=degree_values, rotation=0)
plt.yticks(ticks=np.arange(len(c_values)) + 0.5, labels=c_values, rotation=0)
plt.tight_layout()
plt.show()

k_values = [1, 3, 10, 15, 20, 30, 50, 70, 100]
scores_by_k = {}
for k in k_values:
    knn_model = KNeighborsClassifier(n_neighbors=k, weights='distance')
    auc_scores = cross_val_score(knn_model, x_train, y_train, cv=5, scoring='roc_auc')
    scores_by_k[k] = np.array(auc_scores)

means = [scores.mean() for scores in scores_by_k.values()]
stds = [scores.std() for scores in scores_by_k.values()]

plt.figure(figsize=(8, 6))
main_color = '#1f77b4'
error_color = '#999999'
plt.errorbar(
    k_values, means, yerr=stds,
    fmt='o-',
    capsize=5,
    elinewidth=1.5, capthick=1.5,
    color=main_color, ecolor=error_color,
    label='Mean AUC ± 1 SD'
)

plt.xlabel('K', fontsize=12)
plt.ylabel('Mean ROC_AUC Score', fontsize=12)
plt.title('K-Nearest-Neighbours Cross-Validation for K', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.legend(loc='best')
plt.show()

pf = PolynomialFeatures(degree=6, include_bias=False)
x_train_pl = pf.fit_transform(x_train)
x_test_pl = pf.fit_transform(x_test)

lr_model = LogisticRegression(C=20, penalty='l2')
lr_model.fit(x_train_pl, y_train)
lr_predictions = lr_model.predict(x_test_pl)
intercept, coefficients = lr_model.intercept_[0], lr_model.coef_[0]

print(f'intercept: {intercept}, weights: {coefficients}\n')
print(f'Logistic Regression AUC-Score: {roc_auc_score(y_test, lr_predictions)}')
print(f'Logistic Regression F1-Score: {f1_score(y_test, lr_predictions)}')

knn_model = KNeighborsClassifier(n_neighbors=70, weights='distance')
knn_model.fit(x_train, y_train)
knn_predictions = knn_model.predict(x_test)

print(f'KNN AUC-Score: {roc_auc_score(y_test, knn_predictions)}')
print(f'KNN F1-Score: {f1_score(y_test, knn_predictions)}')

mc_model = DummyClassifier(strategy='most_frequent')
mc_model.fit(x_train, y_train)
mc_predictions = mc_model.predict(x_test)

print(f'Most Common AUC-Score: {roc_auc_score(y_test, mc_predictions)}')
print(f'Most Common F1-Score: {f1_score(y_test, mc_predictions)}\n')

random_model = DummyClassifier(strategy='uniform')
random_model.fit(x_train, y_train)
random_predictions = random_model.predict(x_test)

print(f'Random Classifier AUC-Score: {roc_auc_score(y_test, random_predictions)}')
print(f'Random Classifier F1-Score: {f1_score(y_test, random_predictions)}')

lr_cm = confusion_matrix(y_test, lr_predictions)
print(lr_cm)
disp = ConfusionMatrixDisplay(confusion_matrix=lr_cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix: Logistic Regression")

knn_cm = confusion_matrix(y_test, knn_predictions)
print(knn_cm)
disp = ConfusionMatrixDisplay(confusion_matrix=knn_cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix: K-Nearest Neighbors")

mc_cm = confusion_matrix(y_test, mc_predictions)
print(mc_cm)
disp = ConfusionMatrixDisplay(confusion_matrix=mc_cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix: Most Common Classifier")

random_cm = confusion_matrix(y_test, random_predictions)
print(random_cm)
disp = ConfusionMatrixDisplay(confusion_matrix=random_cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix: Random Classifier")

plt.show()

plt.plot([0, 1], [0, 1], 'k--', lw=1)

lr_proba = lr_model.predict_proba(x_test_pl)[:, 1]
knn_proba = knn_model.predict_proba(x_test)[:, 1]
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_proba)
knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_proba)
plt.plot(lr_fpr, lr_tpr, color='blue', lw=2.5, label='Logistic Regression')
plt.plot(knn_fpr, knn_tpr, color='orange', lw=2.5, label='KNN')

fpr_tpr = lambda tn, fp, fn, tp: [fp / (fp + tn), tp / (tp + fn)]
plt.plot(*fpr_tpr(*random_cm.ravel()), 'go', label='Random Classifier')
plt.plot(*fpr_tpr(*mc_cm.ravel()), 'ro', label='Most Common Classifier')

plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.legend(loc='best')
plt.title('ROC Curve Comparison')

plt.show()

grid_range = np.linspace(-1, 1, 100)
x1_coords, x2_coords = np.meshgrid(grid_range, grid_range)

grid_pred = knn_model.predict(np.column_stack([x1_coords.ravel(), x2_coords.ravel()]))
grid_pred = grid_pred.reshape(x1_coords.shape)

plt.figure(figsize=(8, 6))

plt.contour(x1_coords, x2_coords, grid_pred, alpha=1, color='black')

plt.scatter(y=df[df['y'] == 1]['x2'], x=df[df['y'] == 1]['x1'], color='blue', marker='+', label='y = +1')
plt.scatter(x=df[df['y'] == -1]['x1'], y=df[df['y'] == -1]['x2'], color='orange', marker='o', label='y = -1')

plt.title(f'KNN Decision Boundary k=70', fontsize=14)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.rcParams.update({
    'font.size': 13,
    'axes.labelweight': 'bold',
    'axes.edgecolor': '#333333',
})

plt.show()

