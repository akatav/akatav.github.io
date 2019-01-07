---
layout: page
title: "Existing implementations of FDA"
---

# Implementations

[DiscriMiner](references.html/DiscriMiner) is an R package that implements Fisher's discriminants. We will discuss the code in this package in some detail and use it to classify the IRIS and MNIST datasets. 


Below, we use the [IRIS](references.html/iris) dataset. The data lies in $$R^4$$ and K=3. This means that the projected data points will lie in $$min(K-1,d)=2$$ dimensions. We use the DiscriMiner package to find the covariances. 

Once projected, we visualize the projected data points as seen in the plot below. _FDA_ does a great job with the IRIS dataset with a 97% accuracy. The misclassified data points also can be seen clearly in the projected space as shown in the plot below.

{% highlight r %}
library(DiscriMiner)
K=3 # There are 3 ouput classes, setosa, virginica, versicolor
d=4 # There are 4 features, petal length&width, sepal length&width
N=150 #Number of samples
d1=min(K-1,d)
# Scatter matrices
Sb <- betweenCov(variables = iris[,1:4], group = iris$Species)
Sw <- withinCov(variables = iris[,1:4], group = iris$Species)
{% endhighlight %}

{% highlight r %}
# Eigen decomposition
ev=eigen(solve(Sw)%*%Sb)$vectors
# Get the first d1 eigen vectors
w=ev[,1:d1]
{% endhighlight %}

{% highlight r %}
data=matrix(unlist(iris[,1:4]), nrow=150, ncol=4)
w=matrix(as.numeric(w), nrow=4, ncol=2)
species=as.numeric((iris$Species))
{% endhighlight %}

{% highlight r %}
# Get an orthogonal projection of original data
projecteddata=data%*%w
globalmean=colMeans(iris[,1:4])
groupmeans=groupMeans(iris[,1:4], iris$Species)
projectedgroupmeans=t(groupmeans)%*%w
{% endhighlight %}

{% highlight r %}
#Apply nearest neighbors to test data
correct=0
 for (i in 1:nrow(projecteddata)) {
    projectedpoint=t(replicate(3,projecteddata[i,]))
    closestclass=which.min(sqrt(rowSums((projectedpoint-projectedgroupmeans)^2)))
    print(closestclass)
    if(species[i]==closestclass) {
      correct=correct+1
    }
  }
accuracy=correct/N
{% endhighlight %}

{% highlight r %}
# Plot the projected data points
p=plot(projecteddata[,1], projecteddata[,2], col=species, pch='*')
points(projecteddata[,1], projecteddata[,2], col=testlabels, pch='o')
{% endhighlight %}

![Fisher's projections](/images/Fishersprojections.png)

<a class="continue" href="chapter6.html">Next chapter</a>

