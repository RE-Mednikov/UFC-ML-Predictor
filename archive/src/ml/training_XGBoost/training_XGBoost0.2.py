from xgboost import XGBClassifier
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
import pandas as pd

#define the data
train_df = pd.read_csv("data/features/features_0.2/train.csv")
valuation_df = pd.read_csv("data/features/features_0.2/validation.csv")
test_df = pd.read_csv("data/features/features_0.2/test.csv")


#separate features and target variable
y_train = train_df["fighter_a_won"]
x_train = train_df.drop(columns=["fighter_a_won", "fight_id", "event_id", "event_date", "event_name", "fighter_a_id", "fighter_a_name", "fighter_b_id", "fighter_b_name"])

y_val = valuation_df["fighter_a_won"]
x_val = valuation_df.drop(columns=["fighter_a_won", "fight_id", "event_id", "event_date", "event_name", "fighter_a_id", "fighter_a_name", "fighter_b_id", "fighter_b_name"])

#train the model
model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0,
    reg_alpha=0,
    reg_lambda=1,
    eval_metric="logloss",
    early_stopping_rounds=10,
    random_state=42
)

#train model on train data, test against valuation data and use early stopping if validation prefromance does not improve for 10 rounds (stops training), Verbose = False: hides the per-round training output to keep console cleaner.
model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)


#get results against validation
val_probability = model.predict_proba(x_val)[:, 1]  # Probability of fighter_a winning
val_predictions = (val_probability >= 0.5).astype(int)  # Convert probabilities to binary predictions

print("Validation Log Loss:", log_loss(y_val, val_probability))
print("Validation AUC:", roc_auc_score(y_val, val_probability))
print("Validation Accuracy:", accuracy_score(y_val, val_predictions))