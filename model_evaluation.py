from sklearn import metrics

def model_metrics(y_test, y_prediction, model):

  # Model Accuracy, how often is the classifier correct?
  print(f"{model} Accuracy:",metrics.accuracy_score(y_test, y_prediction))
  print("\n")

  # confusion matrix
  cm= metrics.confusion_matrix(y_test, y_prediction)
  cm_dis = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
  cm_dis.plot()
  #plt.show()

  classx= metrics.classification_report(y_test, y_prediction)
  print("\n")
  print(f"{model} Classification Report:\n", classx)