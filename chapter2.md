---
layout: page
title: "kidding"
---

# A Visual intuition for Fisher's discriminant method in $$R^2$$

Let us plot a set of data points in 2 dimensions using the _R_ code below: 

{% highlight r %}
library(ggplot2)
x <- c(1,-1,0.75,-2,2,2.5,3,0,1,1.5,4,2)
y <- c(1,0.5,-1,-2,1.5,2.4,1.8,4,4,4.2,4, 4.5)
df <- data.frame(cbind(x, y))
colors= c(0,0,0,0,1,1,1,1,2,2,2,2)
g <- ggplot(data = df, aes(x = df$x, y = df$y, colour=factor(colors))) + geom_point()
g <- g + geom_point(alpha = 1/3)  # alpha b/c of overplotting
g <- g + coord_fixed()

g1=g+geom_segment(aes(x=-1,y=5,xend=4,yend=5), arrow=arrow(length = unit(0.03, "npc")))
#Orthogonal projection
ss2<-perp.segment.coord(df$x, df$y, 5, 0)
g1 + geom_segment(data=as.data.frame(ss2), aes(x = x0, y = y0, xend = x1, yend = y1), colour = "black")
{% endhighlight %}

![Fishers discriminant - 1](/images/Fig2.png)

Given a dataset in 2D like above, Fisher's discriminant analysis tries to _find_ a line (or discriminant) onto which the data points are projected orthogonally. Once projected, the projected data points should be linearly seperable. When data points are projected orthogonally on the line in the above plot, we see that the projected points are not well-seperated. In contrast, see plot below for the same set of points. The 3 groups of points are well-seperated on the discriminant. 

![Fishers discriminant - 2](/images/Fig3.png)

To summarize, Fisher's discriminant analysis aims to:

> 1. Find the Fisher's discriminant(s) _L_ for a given dataset _X_
> 2. Orthogonally project the original dataset _X_ onto _L_
> 3. Seperate the data points on _L_ belonging to the _K_ different classes using a threshold. 

Before going any further, let us also discuss the number of Fisher's discriminants needed as _K_ increases. 

<a class="continue" href="chapter3.html">Next chapter</a>

