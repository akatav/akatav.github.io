---
title: "PCA Vs FDA - Data plot"
output:
  html_document:
    df_print: paged
  word_document: default
---

    library(ggbiplot)

    data(iris)
    data=iris[!(iris$Species=="versicolor"),] #without setosa
    pca.obj <- prcomp(data[,c(2,4)],center=TRUE,scale.=TRUE)
    pca.df <- data.frame(Species=data$Species, as.data.frame(pca.obj$x))

    P <- ggbiplot(pca.obj,
             obs.scale = 1, 
             var.scale=1,
             ellipse=T,
             circle=F,
             varname.size=3,
             groups=data$Species)
    P

![](/home/venkat/connectomes/blog/akatav.github.io/_posts/pca-fda_files/figure-markdown_strict/unnamed-chunk-59-1.png)
