
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor, AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from exploratory_data_analysis import X_train, X_test, y_test, y_train, features
import numpy as np



#Functions to print results

r2_dict = {}
def print_evaluate_test(regression_name, true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = abs(metrics.r2_score(true, predicted))
    r2_dict[regression_name] = r2_square
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)
    print('__________________________________')

def print_evaluate_train(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)
    print('__________________________________')




#Linear Regression

lin_reg=LinearRegression()
lin_reg.fit(X_train, y_train)
test_pred = lin_reg.predict(X_test)
train_pred = lin_reg.predict(X_train)
print('\n')
print('\n')
print('LINEAR REGRESSION')
print('Test set evaluation:\n_____________________________________')
print_evaluate_test('Linear Regression', y_test, test_pred)
print('Train set evaluation:\n_____________________________________')
print_evaluate_train(y_train, train_pred)




#Polynomial Regression  

poly_reg = PolynomialFeatures(degree=2)
X_train_2_d = poly_reg.fit_transform(X_train)
X_test_2_d = poly_reg.transform(X_test)

lin_reg = LinearRegression(normalize=True)
lin_reg.fit(X_train_2_d,y_train)

test_pred = lin_reg.predict(X_test_2_d)
train_pred = lin_reg.predict(X_train_2_d)
print('\n')
print('\n')
print('POLYNOMIAL REGRESSION')
print('Test set evaluation:\n_____________________________________')
print_evaluate_test('Polynomial Regression', y_test, test_pred)
print('====================================')
print('Train set evaluation:\n_____________________________________')
print_evaluate_train(y_train, train_pred)




#Ridge Regression 

rg = Ridge()
parameters = {
    "alpha":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
    "normalize":[True,False]}

rg_cv = GridSearchCV(rg, parameters, cv=5)
rg_cv.fit(X_train[features], y_train.values.ravel())
print('\n')
print('\n')
print('RIDGE REGRESSION')
rg_cv.best_estimator_
test_pred = rg_cv.predict(X_test)
train_pred = rg_cv.predict(X_train)
print('Test set evaluation:\n_____________________________________')
print_evaluate_test('Ridge Regression', y_test, test_pred)
print('====================================')
print('Train set evaluation:\n_____________________________________')
print_evaluate_train(y_train, train_pred)




#Lasso Regression

las  = Lasso(alpha=0.1, 
              precompute=True, 
              positive=True, 
              selection='random',
              random_state=42)
las.fit(X_train, y_train)

test_pred = las.predict(X_test)
train_pred = las.predict(X_train)
print('\n')
print('\n')
print('LASSO REGRESSION')
print('Test set evaluation:\n_____________________________________')
print_evaluate_test('Lasso Regression', y_test, test_pred)
print('====================================')
print('Train set evaluation:\n_____________________________________')
print_evaluate_train(y_train, train_pred)




#ElasticNet Regression 

el = ElasticNet(alpha=0.1, l1_ratio=0.9, selection='random', random_state=42)
el.fit(X_train, y_train)

test_pred = el.predict(X_test)
train_pred = el.predict(X_train)
print('\n')
print('\n')
print('ELASTICNET REGRESSION')
print('Test set evaluation:\n_____________________________________')
print_evaluate_test('ElasticNet Regression', y_test, test_pred)
print('====================================')
print('Train set evaluation:\n_____________________________________')
print_evaluate_train(y_train, train_pred)




#Decision Tree Regression

dt = DecisionTreeRegressor()
parameters = {
    "criterion":["squared_error","friedman_mse","absolute_error"],
    "max_depth":[2,4,8,16,32],
    "min_samples_leaf":[2,4,8,16,32],
    "min_samples_split":[2,4,8,16,32]
}

dt_cv = GridSearchCV(dt, parameters, cv=5)
dt_cv.fit(X_train[features],y_train.values.ravel())
dt_cv.best_estimator_
print('\n')
print('\n')
print('DECISION TREE REGRESSION')
test_pred = dt_cv.predict(X_test)
train_pred = dt_cv.predict(X_train)
print('Test set evaluation:\n_____________________________________')
print_evaluate_test('Decision Tree Regression', y_test, test_pred)
print('====================================')
print('Train set evaluation:\n_____________________________________')
print_evaluate_train(y_train, train_pred)




#Random Forest Regression

rf = RandomForestRegressor()
parameters = {
    "max_depth":[2,4,8,16,32],
    "n_estimators":[5,50,250,500]}

rf_cv = GridSearchCV(rf, parameters, cv=5)
rf_cv.fit(X_train[features], y_train.values.ravel())
rf_cv.best_estimator_
print('\n')
print('\n')
print('RANDOM FOREST REGRESSION')
test_pred = rf_cv.predict(X_test)
train_pred = rf_cv.predict(X_train)
print('Test set evaluation:\n_____________________________________')
print_evaluate_test('Random Forest Regression', y_test, test_pred)
print('====================================')
print('Train set evaluation:\n_____________________________________')
print_evaluate_train(y_train, train_pred)




#Adaptive Boosting Regression

ada = AdaBoostRegressor()
parameters = {
    "learning_rate":[0.01,0.1,1.0,10,100],
    "loss":["linear","square","exponential"],
    "n_estimators":[5,50,250,500]}

ada_cv = GridSearchCV(ada, parameters, cv=5)
ada_cv.fit(X_train[features], y_train.values.ravel())
ada_cv.best_estimator_
print('\n')
print('\n')
print('ADAPTIVE BOOSTING REGRESSION')
test_pred = ada_cv.predict(X_test)
train_pred = ada_cv.predict(X_train)
print('Test set evaluation:\n_____________________________________')
print_evaluate_test('Adaptive Boosting Regression', y_test, test_pred)
print('====================================')
print('Train set evaluation:\n_____________________________________')
print_evaluate_train(y_train, train_pred)




#Gradient Boosting Regression

gb = GradientBoostingRegressor()
parameters = {
    "learning_rate":[0.01,0.1,1.0,10,100],
    "max_depth":[2,4,8,16,32],
    "n_estimators":[5,50,250,500]}

gb_cv = GridSearchCV(gb, parameters, cv=5)
gb_cv.fit(X_train[features],y_train.values.ravel())
gb_cv.best_estimator_
print('\n')
print('\n')
print('GRADIENT BOOSTING REGRESSION')
test_pred = gb_cv.predict(X_test)
train_pred = gb_cv.predict(X_train)
print('Test set evaluation:\n_____________________________________')
print_evaluate_test('Gradient Boosting Regression', y_test, test_pred)
print('====================================')
print('Train set evaluation:\n_____________________________________')
print_evaluate_train(y_train, train_pred)




#Extreme Gradient Boosting Regression

xgb = XGBRegressor()
parameters = {
    "n_estimators":[5,50,250,500],
    "max_depth":[2,4,8,16,32],
    "learning_rate":[0.01,0.1,1.0,10,100]}

xgb_cv = GridSearchCV(xgb, parameters, cv=5)
xgb_cv.fit(X_train[features], y_train.values.ravel())
xgb_cv.best_estimator_
print('\n')
print('\n')
print('EXTREME GRADIENT REGRESSION')
test_pred = xgb_cv.predict(X_test)
train_pred = xgb_cv.predict(X_train)
print('Test set evaluation:\n_____________________________________')
print_evaluate_test('Extreme Gradient Boosting Regression', y_test, test_pred)
print('====================================')
print('Train set evaluation:\n_____________________________________')
print_evaluate_train(y_train, train_pred)




#Support Vector Regression

svr = SVR()
parameters = {
    "C":[0.001,0.01,0.1,1.0,10,100,1000],
    "kernel":["linear","poly","rbf","sigmoid"]}

svr_cv = GridSearchCV(svr, parameters, cv=5)
svr_cv.fit(X_train[features], y_train.values.ravel())
svr_cv.best_estimator_
print('\n')
print('\n')
print('SUPPORT VECTOR REGRESSION')
test_pred = svr_cv.predict(X_test)
train_pred = svr_cv.predict(X_train)
print('Test set evaluation:\n_____________________________________')
print_evaluate_test('Support Vector Regression', y_test, test_pred)
print('====================================')
print('Train set evaluation:\n_____________________________________')
print_evaluate_train(y_train, train_pred)




#Stacking Regression

estimators = [("rg",rg_cv.best_estimator_),
              ("dt",dt_cv.best_estimator_),
              ("rf",rf_cv.best_estimator_),
              ("ada",ada_cv.best_estimator_),
              ("gb",gb_cv.best_estimator_),
              ("xgb",xgb_cv.best_estimator_),
              ("svr",svr_cv.best_estimator_)]

sr = StackingRegressor(estimators=estimators)
parameters = {
    "passthrough":[True,False]}

sr_cv = GridSearchCV(sr, parameters, cv=5)
sr_cv.fit(X_train[features],y_train.values.ravel())
sr_cv.best_estimator_
print('\n')
print('\n')
print('STACKING REGRESSION')
test_pred = sr_cv.predict(X_test)
train_pred = sr_cv.predict(X_train)
print('Test set evaluation:\n_____________________________________')
print_evaluate_test('Stacking Regression', y_test, test_pred)
print('====================================')
print('Train set evaluation:\n_____________________________________')
print_evaluate_train(y_train, train_pred)






