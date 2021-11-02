import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split

# parameters

xgb_params = {
    "eta": 0.1,
    "max_depth": 6,
    "min_child_weight": 1,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "nthread": 8,
    "seed": 1,
    "verbosity": 1,
}
n_splits = 5
output_file = "model.bin"


################################ PreProcessing Data ################################ 

df = pd.read_csv("diabetes.csv")

categorical = ['gender']
## Can be converted to numerical
categorical_numerical = ['polyuria','polydipsia','sudden_weight_loss','weakness','polyphagia','genital_thrush','visual_blurring','itching',
        'irritability','delayed_healing','partial_paresis','muscle_stiffness','alopecia','obesity']
numerical = ['age']

df.columns = df.columns.str.lower().str.replace(" ", "_")
## Change 1/0 to True False
for cat in categorical_numerical:
    df[cat] = (df[cat] == 'Yes').values.astype(int)

## Add this columns to numerical after converting
numerical += categorical_numerical

## Convert outcome into 0/1
df['class'] = (df['class'] == 'Positive').values.astype(int)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

    

################################ Train Function ################################
def train(df_train, y_train, xgb_params=xgb_params):
    dicts = df_train[categorical + numerical].to_dict(orient="records")

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    features = dv.get_feature_names()
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)

    model = xgb.train(xgb_params, dtrain, num_boost_round=65)

    return dv, model


################################ Prediction function ################################
def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient="records")

    X = dv.transform(dicts)
    dX = xgb.DMatrix(X, feature_names=dv.get_feature_names())
    y_pred = model.predict(dX)

    return y_pred


################################ Train And try KFold Cross Validation ################################

print('Validating ...')

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []

fold = 0

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train['class'].values
    y_val = df_val['class'].values

    dv, model = train(df_train, y_train, xgb_params)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

    print(f"auc on fold {fold} is {auc}")
    fold = fold + 1


print("validation results:")
print("%.3f +- %.3f" % (np.mean(scores), np.std(scores)))


# training the final model

print("training the final model")

dv, model = train(df_full_train, df_full_train['class'].values, xgb_params)
y_pred = predict(df_test, dv, model)

y_test = df_test['class'].values
auc = roc_auc_score(y_test, y_pred)

print(f"auc={auc}")


# Save the model

with open(output_file, "wb") as f_out:
    pickle.dump((dv, model), f_out)

print(f"the model is saved to {output_file}")