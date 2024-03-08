from sklearn.ensemble import RandomForestClassifier
import joblib

def model_rfc_train(X_train, y_train):
    # Model using the best random forest parameters. Refer the ipynb file for details of the GridSearchCV
    rfc=RandomForestClassifier(random_state=42, max_features='sqrt', n_estimators= 500, max_depth=8, criterion='entropy')
    rfc.fit(X_train, y_train)

    return rfc

def model_rfc_predict(X_test, model):
    return model.predict(X_test)

def save_model(model):
    joblib.dump(model, "./saved_models/best_model.joblib")