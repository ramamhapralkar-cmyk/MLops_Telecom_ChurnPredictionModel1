from sklearn.metrics import classification_report


def model_evaluate(y_test,y_pred):
    score = classification_report(y_test,y_pred)  #(actual,predicted)
    return score