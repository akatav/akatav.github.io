---
title: "Fisher's Discriminant Analysis - Data plot"
output:
  html_document:
    df_print: paged
  word_document: default
---

We will generate a graphic in the following chunk.

    library(ggplot2)

    Person1 <- c(1,-1,0.75,-2,2,2.5,3,0,1,1.5,4,2)
    Person2 <- c(1,0.5,-1,-2,1.5,2.4,1.8,4,4,4.2,4, 4.5)
    df <- data.frame(cbind(Person1, Person2))

    colors= c(0,0,0,0,1,1,1,1,2,2,2,2)
    person_ggplot <- ggplot(data = df, aes(x = Person1, y = Person2, colour=factor(colors))) + geom_point()

    person_ggplot <- person_ggplot + geom_point(alpha = 1/3)  # alpha b/c of overplotting
    person_ggplot <- person_ggplot + coord_fixed()

    person_ggplot

![My helpful screenshot](/images/show_figure-1.png)
