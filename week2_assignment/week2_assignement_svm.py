import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

df = pd.read_csv('week_2_dataset.csv', skiprows=1, header=None, names=['X1', 'X2', 'Y'])
df.assign

y_positive_df = df[df['Y'] == 1]
y_negative_df = df[df['Y'] == -1]

plt.scatter(x=y_positive_df['X1'], y=y_positive_df['X2'], color='blue', marker='+', label='y = +1')
plt.scatter(x=y_negative_df['X1'], y=y_negative_df['X2'], color='orange', marker='o', label='y = -1')

plt.title('Scatter Plot')
plt.xlabel('X1')
plt.ylabel('X2')

plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
plt.show()

X = df[['X1', 'X2']].to_numpy()
y = df['Y'].to_numpy()

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

c_values = [0.0001, 0.001, 0.01, 1, 10, 25, 50, 100]
svc_models = []
for c in c_values:
    model = LinearSVC(loss='hinge', C=c).fit(x_train, y_train)
    svc_models.append(model)
    theta_0, theta_1, theta_2 = *model.intercept_, *model.coef_[0]
    print(f'C={c} | θ_0={theta_0}, θ_1={theta_1}, θ_2={theta_2}')

for model in svc_models:
    pred = model.predict(x_test)
    theta_0, theta_1, theta_2 = *model.intercept_, *model.coef_[0]
    x2 = lambda x1, offset: -x1 * theta_1 / theta_2 - (theta_0 + offset) / theta_2

    train_pos, train_neg = x_train[y_train == 1], x_train[y_train == -1]
    test_pos, test_neg = x_test[pred == 1], x_test[pred == -1]

    plt.scatter(x=train_pos[:, 0], y=train_pos[:, 1], color='blue', marker='+', label='y_train = +1')
    plt.scatter(x=train_neg[:, 0], y=train_neg[:, 1], color='orange', marker='o', label='y_train = -1')

    plt.scatter(x=test_pos[:, 0], y=test_pos[:, 1], marker='+', color='red', label='y_pred = +1')
    plt.scatter(x=test_neg[:, 0], y=test_neg[:, 1], marker='o', color='green', label='y_pred = -1')

    plt.plot([-1, 1], [x2(-1, 0), x2(1, 0)], color='black', linewidth='2.5', label='Decision Boundary')
    plt.plot([-1, 1], [x2(-1, 1), x2(1, 1)], '--', color='black', linewidth='1.5', label='Margin +1', )
    plt.plot([-1, 1], [x2(-1, -1), x2(1, -1)], '--', color='black', linewidth='1.5', label='Margin -1')


    plt.ylim(-1.1, 1.1)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(f'C: {model.C}, Boundary: {round(theta_2, 2)}x_2 + {round(theta_1, 2)}x_1 + {round(theta_0, 2)} = 0')
    plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1))
    plt.show()
    print(classification_report(y_test, pred))
