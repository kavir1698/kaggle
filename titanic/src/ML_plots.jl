using PyCall
using PyPlot
using ScikitLearn
@pyimport scipy
interp = scipy.interp
@sk_import metrics: roc_curve
@sk_import metrics: auc
@sk_import model_selection: StratifiedKFold
@sk_import model_selection: learning_curve  
@sk_import model_selection: validation_curve  
@sk_import metrics: precision_recall_curve
@sk_import metrics: average_precision_score
@sk_import model_selection: cross_val_predict


"""
  Validation curve.

  Determine training and test scores for varying parameter values.

  Compute scores for an estimator with different values of a specified parameter. This is similar to grid search with one parameter. However, this will also compute training scores and is merely a utility for plotting the results.
"""
function plot_valid_curve(estimator, X, y, param_name, param_range; cv=5, scoring=nothing)
  train_scores, test_scores = validation_curve(estimator, X, y, param_name=param_name, param_range=param_range, cv=cv, scoring=scoring, n_jobs=4)
  train_scores_mean = mean(train_scores, 2)
  train_scores_std = std(train_scores, 2)
  test_scores_mean = mean(test_scores, 2)
  test_scores_std = std(test_scores, 2)

  title("Validation Curve")
  xlabel(param_name)
  ylabel("Score")
  ylim(0.0, 1.1)
  lw = 2
  semilogx(param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw)

  pn = Array{Float64, 1}(0)
  for i in (train_scores_mean - train_scores_std)
    push!(pn, i)
  end
  pp = Array{Float64, 1}(0)
  for i in (train_scores_mean + train_scores_std)
    push!(pp, i)
  end
  fill_between(param_range, pn, pp, alpha=0.2, color="darkorange", lw=lw)

  semilogx(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw)

  pn = Array{Float64, 1}(0)
  for i in (test_scores_mean - test_scores_std)
    push!(pn, i)
  end
  pp = Array{Float64, 1}(0)
  for i in (test_scores_mean + test_scores_std)
    push!(pp, i)
  end
  fill_between(param_range, pn, pp, alpha=0.2, color="navy", lw=lw)
  legend(loc="best")
  savefig("validation_curve.pdf", dpi=300)
  close()
end

function plot_cv_preds(estimator, X, y)
  predicted = cross_val_predict(estimator, X, y, cv=10)
  fig, ax = subplots()
  scatter(y, predicted, edgecolors=(0, 0, 0))
  plot([minimum(y), maximum(y)], [minimum(y), maximum(y)], "k--", lw=4)
  ax[:set_xlabel]("Measured")
  ax[:set_ylabel]("Predicted")
  savefig("crossval_predictions.pdf", dpi=300)
  close()
end


function plot_precision_recall(estimator, X_test, y_test)
  y_score = estimator[:decision_function](X_test)
  precision, recall, jj = precision_recall_curve(y_test, y_score)
  average_precision = average_precision_score(y_test, y_score)

  step(recall, precision, color="b", alpha=0.2, where="post")
  fill_between(recall, precision, step="post", alpha=0.2, color="b")
  xlabel("Recall")
  ylabel("Precision")
  ylim([0.0, 1.05])
  xlim([0.0, 1.0])
  title("2-class Precision-Recall curve: AP=$(round(average_precision, 2))")
  savefig("precisionRecallCurve.pdf", dpi=300)
  close()
end

function plot_roc(estimator, X, y)
  est = deepcopy(estimator)
  tprs = []
  aucs = []
  mean_fpr = linspace(0.01, 1, 100)
  cv = StratifiedKFold(n_splits=6)
  i = 0
  for (train, test) in cv[:split](X, y)
    train = train + 1
    test = test + 1
    probas_ = fit!(est, X[train, :], y[train])[:predict_proba](X[test, :])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test, :], probas_[:, 1])
    push!(tprs, interp(mean_fpr, fpr, tpr))
    tprs[end][1] = 0.0
    roc_auc = auc(fpr, tpr)
    push!(aucs, roc_auc)
    plot(fpr, tpr, lw=1, alpha=0.3, label="ROC fold $i (AUC = $(round(roc_auc, 2)))")
    i += 1
  end

  plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Luck", alpha=.8)

  mean_tpr = mean(tprs)
  mean_tpr[end] = 1.0
  mean_auc = auc(mean_fpr, mean_tpr)
  std_auc = std(Array{Float64}(aucs))
  plot(mean_fpr, mean_tpr, color="b", label= "Mean ROC (AUC = $(round(mean_auc, 2)) +- $(round(std_auc, 2))", lw=2, alpha=0.8)

  tprs_columns = []
  for i in 1:length(tprs[1])
    push!(tprs_columns, [j[i] for j in tprs])
  end

  std_tpr = [std(i) for i in tprs_columns]
  tprs_upper = minimum(mean_tpr + std_tpr)[1]
  tprs_lower = maximum(mean_tpr - std_tpr)[1]
  fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey", alpha=0.2, label="+- 1 std. dev.")

  xlim([-0.05, 1.05])
  ylim([-0.05, 1.05])
  xlabel("False Positive Rate")
  ylabel("True Positive Rate")
  title("Receiver operating characteristic curve")
  legend(loc="best")
  savefig("ROC.pdf")
  close()
end

function plot_trainTest_errors(estimator, X, y)
  est = deepcopy(estimator)
  alphas = logspace(-5, 3, 60)
  train_errors = []
  test_errors = []
  for alpha in alphas
      set_params!(est, C=alpha)
      fit!(est, X_train, y_train)
      push!(train_errors, est[:score](X_train, y_train))
      push!(test_errors, est[:score](X_test, y_test))
  end

  i_alpha_optim = findin(test_errors, maximum(test_errors))[1]
  alpha_optim = alphas[i_alpha_optim]
  print("Optimal regularization parameter : $alpha_optim")

  # Estimate the coef_ on full data with optimal regularization parameter
  set_params!(est, C=alpha_optim)
  coef_ = fit!(est, X, y)[:coef_]

  # ###
  # Plot results functions
  # subplot(2, 1, 1)
  semilogx(alphas, train_errors, label="Train")
  semilogx(alphas, test_errors, label="Test")
  vlines(alpha_optim, ylim()[1], maximum(test_errors), color="k", linewidth=3, label="Optimum on test")
  legend(loc="top left")
  ylim([0, 1.2])
  xlabel("Regularization parameter")
  ylabel("Performance")
  savefig("testTrainErros.pdf", dpi=300)
  close()
end


"""
  Generate a simple plot of the test and training learning curve.

  Parameters
  ----------
  estimator : object type that implements the "fit" and "predict" methods
      An object of that type which is cloned for each validation.

  title : string
      Title for the chart.

  X : array-like, shape (n_samples, n_features)
      Training vector, where n_samples is the number of samples and
      n_features is the number of features.

  y : array-like, shape (n_samples) or (n_samples, n_features), optional
      Target relative to X for classification or regression'
      None for unsupervised learning.

  ylim : tuple, shape (ymin, ymax), optional
      Defines minimum and maximum yvalues plotted.

  cv : int, cross-validation generator or an iterable, optional
      Determines the cross-validation splitting strategy.
      Possible inputs for cv are:
        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.

      For integer/None inputs, if ``y`` is binary or multiclass,
      :class:`StratifiedKFold` used. If the estimator is not a classifier
      or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

      Refer :ref:`User Guide <cross_validation>` for the various
      cross-validators that can be used here.

  n_jobs : integer, optional
      Number of jobs to run in parallel (default 1).

  scoring: the scoring of algorithm. default nothing. 
"""
function plot_learning_curve(estimator, X, y; pcax=[0.0 0.0], gtitle="Learning Curves", train_sizes=0.18:0.05:1.0, cv=10, scoring=nothing)
  figure()
  title(gtitle)
  xlabel("Training examples")
  ylabel(scoring)
  train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, train_sizes=train_sizes, scoring=scoring)
  if scoring == nothing
    ylabel("Score")
  end
  train_scores_mean = mean(train_scores, 2)
  train_scores_std = std(train_scores, 2)
  test_scores_mean = mean(test_scores, 2)
  test_scores_std = std(test_scores, 2)
  grid()

  y1 = Array{Float64}(0)
  j = train_scores_mean - train_scores_std
  for i in j
    push!(y1, i)
  end
  y2 = Array{Float64}(0)
  j = train_scores_mean + train_scores_std
  for i in j
    push!(y2, i)
  end
  fill_between(train_sizes, y1, y2, alpha=0.1, color="r")

  y1 = Array{Float64}(0)
  j = test_scores_mean - test_scores_std
  for i in j
    push!(y1, i)
  end
  y2 = Array{Float64}(0)
  j = test_scores_mean + test_scores_std
  for i in j
    push!(y2, i)
  end
  fill_between(train_sizes, y1, y2, alpha=0.1, color="g")
  plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
  plot(train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score")

  legend(loc="best")
  savefig("learning_curve.pdf", dpi=300)
  close()
end