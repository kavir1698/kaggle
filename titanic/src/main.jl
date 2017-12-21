using DataFrames
using PyPlot
using ScikitLearn
using ScikitLearn: fit!, predict, transform
include("ML_plots.jl")
# using PyCall, JLD, PyCallJLD

## read the data
data_file = "../data/train.csv"
df = readtable(data_file, separator=',', header=true)
test_data_file = "../data/test.csv"
df_test = readtable(test_data_file, separator=',', header=true)


# 0. prepare the data
describe(df)
describe(df_test)
parameter_names = DataFrames.names(df)[[1, 3, 5, 6, 7, 8, 10, 12]]
# parameter_names = DataFrames.names(df)[[3, 5, 6]]
for pname in parameter_names
  println("Number of missing $pname in train data: $(length(find(a -> isna(a), df[pname])))")
  println("Number of missing $pname in test data: $(length(find(a -> isna(a), df_test[pname])))")
end


## Adding titles column
pnames = df[:Name]
titles = []
for nn in pnames
  first = split(nn, ",")[2]
  title = split(first, ".")[1]
  title = strip(title)
  push!(titles, title)
end
unique(titles)
title_indices = Dict()
for (index, tt) in enumerate(unique(titles))
  title_indices[tt] = index
end

df[:Title] = zeros(1:size(df)[1])
for rr in 1:size(df)[1]
  fullname = df[rr, :Name]
  firstname = split(fullname, ",")[2]
  title = split(firstname, ".")[1]
  title = strip(title)
  if haskey(title_indices, title)
    df[rr, :Title] = title_indices[title]
  else
    title_indices[title] = length(keys(title_indices)) + 1
    df[rr, :Title] = title_indices[title]
  end
end
push!(parameter_names, :Title)

df_test[:Title] = zeros(1:size(df_test)[1])
for rr in 1:size(df_test)[1]
  fullname = df_test[rr, :Name]
  firstname = split(fullname, ",")[2]
  title = split(firstname, ".")[1]
  title = strip(title)
  if haskey(title_indices, title)
    df_test[rr, :Title] = title_indices[title]
  else
    title_indices[title] = length(keys(title_indices)) + 1
    df_test[rr, :Title] = title_indices[title]
  end
end

## add FamilySize column
df[:FamilySize] = df[:SibSp] + df[:Parch]
push!(parameter_names, :FamilySize)

df_test[:FamilySize] = df_test[:SibSp] + df_test[:Parch]

## add isAlone column
alones = ones(1:size(df)[1])
not_alones = find(a -> a>0, df[:FamilySize])
alones[not_alones] = 0
df[:isAlone] = alones

push!(parameter_names, :isAlone)

alones = ones(1:size(df_test)[1])
not_alones = find(a -> a>0, df_test[:FamilySize])
alones[not_alones] = 0
df_test[:isAlone] = alones

## fill in the missing values
# @sk_import preprocessing: Imputer

## Add missing ages using total median
# missing_ages = find(a -> isna(a), df[:Age])
# df[:Age][missing_ages] = 999
# im = Imputer(missing_values=999, strategy="median", axis=0)
# fit!(im, DataArray(df[[:Age, :SibSp]]))
# j = transform(im, DataArray(df[[:Age, :SibSp]]))
# df[:Age] = j[:, 1]

## Add missing ages using the median within the same sex and title
missing_ages = find(a -> isna(a), df[:Age])
for rr in missing_ages
  df[rr, :Age] = median(df[.&(~isna(df[:Age]), df[:Sex] .== df[rr, :Sex], df[:Title] .== df[rr, :Title]), :Age])
end

missing_embarked = find(a -> isna(a), df[:Embarked])
el = levels(df[:Embarked])
el_freq = []
for ii in el
  push!(el_freq, length(df[.&(~isna.(df[:Embarked]), df[:Embarked] .== ii), 12]))
end
most_freq = el[findin(el_freq, maximum(el_freq))][1]

df[:Embarked][missing_embarked] = most_freq


### filling in missing test data
## Add missing ages using total median
# missing_ages = find(a -> isna(a), df_test[:Age])
# df_test[:Age][missing_ages] = 999
# im = Imputer(missing_values=999, strategy="median", axis=0)
# fit!(im, DataArray(df_test[[:Age, :SibSp]]))
# j = transform(im, DataArray(df_test[[:Age, :SibSp]]))
# df_test[:Age] = j[:, 1]

## Add missing ages using the median within the same sex and title
missing_ages = find(a -> isna(a), df_test[:Age])
for rr in missing_ages
  try
    df_test[rr, :Age] = median(df_test[.&(~isna(df_test[:Age]), df_test[:Sex] .== df_test[rr, :Sex], df_test[:Title] .== df_test[rr, :Title]), :Age])
  catch
    df_test[rr, :Age] = median(df_test[.&(~isna(df_test[:Age]), df_test[:Sex] .== df_test[rr, :Sex], df_test[:Title] .== df_test[rr, :Pclass]), :Age])
  end
end

missing_fares = find(a -> isna(a), df_test[:Fare])
df_test[:Fare][missing_fares] = 9999
im = Imputer(missing_values=9999, strategy="median", axis=0)
fit!(im, DataArray(df_test[[:Fare, :SibSp]]))
j = transform(im, DataArray(df_test[[:Fare, :SibSp]]))
df_test[:Fare] = j[:, 1]



X = Matrix(df[:, parameter_names[2:end]])
X_test = Matrix(df_test[:, parameter_names[2:end]])
nsamples = size(X)[1]
nsamples_test = size(X_test)[1]
y = Array(df[:Survived])

# replace strings in X with integers
string_columns = [2, 7]
transforming_dicts = []
for cc in string_columns
  sl = levels(X[:, cc])
  new_levels = 1:length(sl)
  new_levels_dict = Dict()
  for index in new_levels
    new_levels_dict[sl[index]] = index
  end
  for rr in 1:nsamples
    X[rr, cc] = new_levels_dict[X[rr, cc]]
  end
  for rr in 1:nsamples_test
    X_test[rr, cc] = new_levels_dict[X_test[rr, cc]]
  end
  push!(transforming_dicts, new_levels_dict)
end

### PCA
@sk_import decomposition: PCA
function apply_pca(X, parameter_names)
  pca = PCA()
  fit!(pca, X)
  # explained variances
  all_v = pca[:explained_variance_]
  println("Explained variance by each feature:")
  for (index, nn) in enumerate(parameter_names)
    println("$nn: ", all_v[index])
  end
  xp = X
  for i in 1:length(parameter_names)
    pca[:n_components] = i
    fit!(pca, X)
    if sum(pca[:explained_variance_ratio_]) > 0.999
      xp = fit_transform!(pca, X)
      break
    end
  end
  return xp, pca
end
Xpca, pca = apply_pca(X, parameter_names[2:end])

# normalize features
# @sk_import preprocessing: RobustScaler
@sk_import preprocessing: MinMaxScaler
rscale = MinMaxScaler()
XpcaScaled = rscale[:fit_transform](Xpca)
Xscaled = rscale[:fit_transform](X)

##################################################
### A logistic regression to predict survivals ###
##################################################

@sk_import linear_model: LogisticRegression
@sk_import model_selection: GridSearchCV
@sk_import model_selection: train_test_split
@sk_import model_selection: KFold

# 1. build the model
parameters = Dict("penalty"=> ("l1", "l2"), "C" => (0.001, 0.01, 0.1, 0.4, 1.0, 2, 4, 10, 44, 55, 100, 1000))
lreg = LogisticRegression(n_jobs=4, random_state=3)
kf = KFold(n_splits=10, shuffle=false)
grd_reg = GridSearchCV(lreg, parameters, scoring="accuracy", cv=kf)
fit!(grd_reg, X, y)
# fit!(grd_reg, XpcaScaled, y)
# fit!(grd_reg, Xscaled, y)
# fit!(grd_reg, Xpca, y)
best_estimator = grd_reg[:best_estimator_]

#=
# model results
grd_reg[:best_params_] # l2, 0.1
grd_reg[:best_score_]  # 0.802

=#

# since the predictions are pretty low, I produce polynomials
@sk_import preprocessing: PolynomialFeatures
poly = PolynomialFeatures(4, interaction_only=true)
X_poly2 = fit_transform!(poly, X)
# X_poly2 = fit_transform!(poly, XpcaScaled)
# X_poly2 = fit_transform!(poly, Xpca)
# X_poly2 = fit_transform!(poly, Xscaled)


parameters = Dict("penalty"=> ("l1", "l2"), "C" => (0.001, 0.01, 0.1, 1, 2, 4, 10, 44, 55, 85, 100, 1000))
lreg = LogisticRegression(n_jobs=4, random_state=3)
kf = KFold(n_splits=10, shuffle=false)
grd_reg = GridSearchCV(lreg, parameters, scoring="accuracy", cv=kf)
fit!(grd_reg, X_poly2, y)
best_estimator = grd_reg[:best_estimator_]

#=
# model results
grd_reg[:best_params_] # l2, 55
grd_reg[:best_score_]  # 0.83

=#

# test the estimator on test data and write to file
transformed_x_test = fit_transform!(poly, X_test)
y_pred = predict(best_estimator, transformed_x_test)
outfile = "../results/logistic_regression_predictions.csv"
f = open(outfile, "w")
println(f, "PassengerId,Survived")
close(f)
open(outfile, "a") do ff
  for index in 1:length(df_test[:PassengerId])
    println(ff, df_test[index, :PassengerId], ",", y_pred[index])
  end
end


#################################
### A SVC to predict survival ###
#################################
@sk_import svm: SVC
@sk_import model_selection: GridSearchCV
@sk_import model_selection: train_test_split
@sk_import model_selection: KFold


# 1. build the model
parameters = [Dict("kernel"=> ["rbf"], "gamma" => ("auto", 0.0001, 0.001, 0.01, 0.1, 1, 10, 44, 55, 100, 1000)), Dict("kernel"=> ["poly"], "gamma" => ("auto", 0.0001, 0.001, 0.01, 0.1, 1, 10, 44, 55, 100, 1000), "degree" => (1,2,3,4,5,6))]
lreg = SVC()
kf = KFold(n_splits=7, shuffle=false)
grd_reg = GridSearchCV(lreg, parameters, scoring="accuracy", cv=kf)
# fit!(grd_reg, X, y)  # TOO SLOW
# fit!(grd_reg, XpcaScaled, y)  # TOO SLOW
# best_estimator = grd_reg[:best_estimator_]

#=
# model results
grd_reg[:best_params_] # kernel= poly, gamma=10, degree = 6
grd_reg[:best_score_]  # 0.98

=#
# test the estimator on test data
y_pred = predict(best_estimator, X_test)
outfile = "../results/svc_predictions.csv"
f = open(outfile, "w")
println(f, "PassengerId,Survived")
close(f)
open(outfile, "a") do ff
  for index in 1:length(df_complete_test[:PassengerId])
    println(ff, df_complete_test[index, :PassengerId], ",", y_pred[index])
  end
end



###########################################
### A random forest to predict survival ###
###########################################
@sk_import ensemble: RandomForestClassifier
@sk_import model_selection: GridSearchCV
@sk_import model_selection: train_test_split
@sk_import model_selection: KFold


parameters = Dict("n_estimators"=> (2, 10, 20, 40, 60, 80, 100), "max_depth"=>(nothing, 2, 4, 6, 8, 10, 12), "max_features" => (3, 4, 5, 6))
rfc = RandomForestClassifier()
kf = KFold(n_splits=10, shuffle=false)
grd_reg = GridSearchCV(rfc, parameters, scoring="accuracy", cv=kf)
fit!(grd_reg, X, y)
# fit!(grd_reg, Xpca, y)
best_estimator = grd_reg[:best_estimator_]

#= 
grd_reg[:best_params_]  # "max_depth"    => 8, "n_estimators" => 80, max_features: 5
grd_reg[:best_score_]  # 0.8496  includig the Title column
best_estimator[:feature_importances_]
=#


# since the predictions are pretty weak, I produce polynomials
@sk_import preprocessing: PolynomialFeatures
poly = PolynomialFeatures(2, interaction_only=false)
X_poly2 = fit_transform!(poly, X)

parameters = Dict("n_estimators"=> (1, 2, 10, 20, 40, 60, 80, 100), "max_depth"=>(nothing, 2, 4, 6, 8, 10, 12, 14), "max_features" => (2, 3, 4, 5, 6))
rfc = RandomForestClassifier()
kf = KFold(n_splits=10, shuffle=false)
grd_reg = GridSearchCV(rfc, parameters, scoring="f1", cv=kf)
fit!(grd_reg, X_poly2, y)
best_estimator = grd_reg[:best_estimator_]
#= 
grd_reg[:best_params_]  # "max_depth"    => 8, "n_estimators" => 100, max_features: 6
grd_reg[:best_score_]  # 0.7845  using the Title column too
=#

# test the estimator on test data
y_pred = predict(best_estimator, X_test)
outfile = "../results/randomForest_predictions.csv"
f = open(outfile, "w")
println(f, "PassengerId,Survived")
close(f)
open(outfile, "a") do ff
  for index in 1:length(df_test[:PassengerId])
    println(ff, df_test[index, :PassengerId], ",", y_pred[index])
  end
end


##########################################################
### A gradient boosting classifier to predict survival ###
##########################################################
@sk_import ensemble: GradientBoostingClassifier
@sk_import model_selection: GridSearchCV
@sk_import model_selection: train_test_split
@sk_import model_selection: KFold



parameters = Dict("learning_rate"=> (0.01, 0.1, 0.2, 0.4, 0.6, 1, 2, 4), "max_depth" => (3,4,5,6), "n_estimators" => (100, 200, 250, 300, 400, 500))
rfr = GradientBoostingClassifier()
kf = KFold(n_splits=15, shuffle=false)
grd_reg = GridSearchCV(rfr, parameters, scoring="accuracy", cv=kf)
fit!(grd_reg, X, y)
best_estimator = grd_reg[:best_estimator_]


# model results
grd_reg[:best_params_]  # "max_depth"    => 5, "learning_rate" => 0.01, n_estimators=300
grd_reg[:best_score_] # 0.845 using Titles
best_estimator[:feature_importances_]



# since the predictions are pretty weak, I produce polynomials
@sk_import preprocessing: PolynomialFeatures
poly = PolynomialFeatures(2, interaction_only=true)
X_poly2 = fit_transform!(poly, X)

parameters = Dict("learning_rate"=> (0.01, 0.1, 0.2, 0.4, 0.6, 1, 2, 4), "max_depth" => (3,4,5,6), "n_estimators" => (100, 200, 250, 300, 400, 500))
rfr = GradientBoostingClassifier()
kf = KFold(n_splits=15, shuffle=false)
grd_reg = GridSearchCV(rfr, parameters, scoring="accuracy", cv=kf)
fit!(grd_reg, X_poly2, y)
best_estimator = grd_reg[:best_estimator_]

# model results
grd_reg[:best_params_]  # "max_depth"    => 5, "learning_rate" => 1, n_estimators=300
grd_reg[:best_score_] # 0.83


## plots
## 1
plot_learning_curve(best_estimator, X, y)

## 3. precision-recall curve
plot_precision_recall(best_estimator, X_test, y_test)

## 4. train-test errors
plot_trainTest_errors(best_estimator, X, y)

## 4. ROC curve
plot_roc(best_estimator, X, y)

## 5. validation curve
plot_valid_curve(best_estimator, X, y, "C", [0.001, 0.01, 0.1, 1, 10, 50, 100, 1000]; cv=5, scoring="f1")


# test the estimator on test data
y_pred = predict(best_estimator, X_test)
outfile = "../results/gbc_predictions.csv"
f = open(outfile, "w")
println(f, "PassengerId,Survived")
close(f)
open(outfile, "a") do ff
  for index in 1:length(df_test[:PassengerId])
    println(ff, df_test[index, :PassengerId], ",", y_pred[index])
  end
end