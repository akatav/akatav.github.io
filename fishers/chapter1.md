---
layout: home
title: "introduction"
---

# Introduction

Fisher's discriminant analysis was first introduced by [Fisher, 1936](../references.html#Fisher1936). Fisher's discriminant analysis is a supervised data analysis technique. This means that it is mandatory that the (training) data samples that are to be classified need to be labelled as belonging to one of the output classes. Fisher's Discriminant Analysis (hitherto referred to as _FDA_) is also a  dimensionality reduction method in statistical data analysis. This is helpful when the data resides in higher dimensions and needs to be projected onto a lower dimension for easy analysis. In this article, we aim to discuss FDA in detail, both mathematically and implementation-wise. 

> Given dataset _X_ in _d_ dimensions where each data sample $$x_i$$ is assigned to one of _K_ discrete output class labels. _FDA_ aims to find a _discriminant_ in a lower dimension $$d' < d$$ onto which _X_ is projected. The projected data points belonging to the _K_ classes are easily seperated on this discriminant. 

### Differences between Fisher's Discriminant analysis and PCA

_PCA_ is Principal Component Analysis. Like _FDA_, _PCA_ is also a dimensionality reduction technique. It aims to find the directions (or principal components) along which data varies the most. It does not, however, require that the data be labelled like _FDA_. Although, this is beyond  the scope of this article, it is important to draw a contrast between _FDA_ and _PCA_ as both these techniques use common techniques from linear algebra. The main difference between _PCA_ and _FDA_ is one of maximum variance and maximum seperability respectively. Let us use a picture aid to understand this briefly. 

Let's plot the _IRIS_ data. Although for purposes of illustration, we will consider only 2 features (_petal length_, _sepal width_) and 2 categories (output labels) namely, _setosa_ and _versicolor_ from the original data. 

{% highlight r %}
library(ggbiplot)
data=iris[!(iris$Species=="versicolor"),] #without setosa
pca.obj <- prcomp(data[,c(2,4)],center=TRUE,scale.=TRUE)
pca.df <- data.frame(Species=data$Species, as.data.frame(pca.obj$x))
P <- ggbiplot(pca.obj,obs.scale = 1,var.scale=1,ellipse=T,circle=F,varname.size=3,groups=data$Species)
P
{% endhighlight %}

![PCA-FDA](/images/Fig1.png)

Note that, PCA find the directions along which the data varies the most indicated by the 2 brown arrows. However, _FDA_ tries to _find_ the line (or discriminant) given by the black arrow. Projecting the data points onto the black arrow seperates the original data well. Also, note that _PCA_ does _not_ require labelled data. It is a dimensionality reduction method that looks at the entire data and find outs the directions of maximum variance. 

We shall come back to the technical differences between the two methods again in Chapters 5 and 6. 

### Core idea and intuition

If there are such discriminant(s) that provide good seperability to the data after projection, then such discriminant(s) will maximize the distance between each group of points while making each individual group as compact as possible. Maximizing seperatedness between different groups of points of different classes is akin to maximizing the distances between the mean of each group of points. Making each group of points compact is akin to minimizing the euclidean distances between points in a group and their respective mean. This is the core idea behind finding a suitable projection of points on Fisher's discriminant(s). 

<a class="continue" href="chapter2.html">Next chapter</a>

