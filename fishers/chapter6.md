---
layout: home
title: "philosopher"
---

# Implementing between and within class covariance

_DiscriMiner_ uses the __betweenCov__ and __withinCov__ methods to find the between class and within class covariance, $$S_b$$ and $$S_w$$ respectively. We've reproduced the DiscriMiner code for these 2 methods below, although comments are our own. 

## DiscriMiner betweenCov
{% highlight r%}
# DiscriMiner code for finding S_b
# I've added comments for my own understanding
function (variables, group, div_by_n = FALSE) 
{
  verify_Xy = my_verify(variables, group, na.rm = FALSE)
  X = verify_Xy$X
  y = verify_Xy$y
  n = nrow(X)
  p = ncol(X) # p - number of dimensions/features
  glevs = levels(y) #Returns setosa,versicolor,virginica namely the output labels
  ng = nlevels(y) #Returns 3, the number of distinct output labels
  mean_global = colMeans(X) #mean of original data - a (p,1) vector of means of sepal len,sepal width,
  # petal len and petal width of the entire data
  Between = matrix(0, p, p) # $$S_b$$ is a zero matrix of dimension {p,p}
  # of between-class covariance
  for (k in 1:ng) { # for each output class
    tmp <- y == glevs[k]
    nk = sum(tmp)
    mean_k = colMeans(X[tmp, ]) # mean of this output class - a (p,1) vector of means of sepal len,sepal width,
    # petal len and petal width of the data in class k
    dif_k = mean_k - mean_global
    if (div_by_n) {
      # euclidean distance between global mean and class mean. R uses the tcrossprod function to these 2 vectors, we can use %*% also instead like
      # dif_k %*% t(dif_k). We can also use crossprod(t(dif_k)).
      # The covariance is weighed against the ratio between number of observations in this class and the total number of observations.
      between_k = (nk/n) * tcrossprod(dif_k)
    }
    else {
      between_k = (nk/(n - 1)) * tcrossprod(dif_k)
    }
    # S_b is the sum of the S_b of data belonging to each output class
    Between = Between + between_k
  }
  if (is.null(colnames(variables))) {
    var_names = paste("X", 1:ncol(X), sep = "")
    dimnames(Between) = list(var_names, var_names)
  }
  else {
    dimnames(Between) = list(colnames(variables), colnames(variables))
  }
  Between
}
{% endhighlight %}

## DiscriMiner withinCov
{% highlight r%}
function (variables, group, div_by_n = FALSE) 
{
  verify_Xy = my_verify(variables, group, na.rm = FALSE)
  X = verify_Xy$X
  y = verify_Xy$y
  n = nrow(X)
  p = ncol(X)
  glevs = levels(y)
  ng = nlevels(y)
  Within = matrix(0, p, p)
  for (k in 1:ng) {
    tmp <- y == glevs[k]
    nk = sum(tmp)
    # var() returns the covariance of data belonging to class k
    # weighed against the number of observations
    # cov() also returns the same matrix

    if (div_by_n) {
      Wk = ((nk - 1)/n) * var(X[tmp, ])
    }
    else {
      # The covariance is weighed against the ratio of the number of observations in class k
      # and the total number of observations
      Wk = ((nk - 1)/(n - 1)) * var(X[tmp, ])
    }
    # $$S_w = S_w1 + S_w2 + ... + S_wk$$
    Within = Within + Wk
  }
  if (is.null(colnames(variables))) {
    var_names = paste("X", 1:ncol(X), sep = "")
    dimnames(Within) = list(var_names, var_names)
  }
  else {
    dimnames(Within) = list(colnames(variables), colnames(variables))
  }
  Within
}
{% endhighlight %}
<a class="continue" href="chapter9.html">Next chapter</a>

