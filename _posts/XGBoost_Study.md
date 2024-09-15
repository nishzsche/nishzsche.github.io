---
layout: post
title: "XGBoost_Study"
---

<a href="https://colab.research.google.com/github/nishzsche/nishzsche.github.io/blob/gh-pages/XGBoost_Study.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


```python

```

1.  Play around with the "gamma" regularisation parameter. Gamma adds a complexity cost to the XGBoost trees. The XGBoost trees will only become deeper if the gain associated with expanding the tree is > gamma (also known as “pruning”). The higher gamma, the stronger the regularisation.
2. You can use the "subsample" parameter to train each XGBoost tree on a subset of the data. If subsample is set to 0.9, each tree will be trained on 90% of the training data.
3. Use the "colsample_bytree" parameter where again you would train each XGBoost tree on a random subset of your features. If you set colsample_bytree = 0.9, you would randomly remove 10% of the features when building each tree.
4. Use early stopping. As you keep adding estimators (i.e trees) to better fit the training data, you will start to overfit. Early_stopping_rounds will stop XGBoost from adding additional trees when its performance on the validation set stops improving for a certain number of trees.
5. Play around with the lambda and alpha regularisation parameters which are similar to L2 and L1 regularisation.

Bonus, like for random forest you can play around with the tree parameters as well:

6. Adjust the “max_depth” parameter of the trees, the lower the max depth, the simpler your algorithm and the stronger the regularisation.

7. Adjust the min_samples_leaf parameter to set a minimum number of samples per leaf. The higher the value the stronger the regularisation.

8. dispensing with it when the dataset is fairly small and relationships are quite simple/linear. That would, in most cases, be fitting a square peg into a round hole.

9. Another non-hyperparameter strategy: bucketing variables to make the algo less able to slice and dice continuous or ordinal variables too much and create overly complex relationships.

10. And the best algo for tabular data is the one you’ve actually tried and cross-validated to be the best by whatever metric makes the most sense for your specific dataset and business goal.

11. XGBoost is not the best one for tabular data, it is just the well-known one. CatBoost outperforms XGBoost in many ways. As you are discussing about overfit, CatBoost hardly overfits and it works perfectly well on numeric features. Of course, althe more complicated a ml model it is, the more time someone needs to master it.

12. The good news is, because of the bagging method w/ rf, overfitting is very hard?

13. I love xgboost, but I find if you are good at parameter tuning (using a good chunk of the techniques you have outlined above) lightgbm performs nearly as well for the huge increase in speed. Further EvoTrees does both better with proper tuning.

14. However I have been resold on simple linear regression recently. If you can modify the features to encode some of the nonlinearities,
Linear Regression (or more properly "geodesic regression" ) outperforms everything.



```python

```
