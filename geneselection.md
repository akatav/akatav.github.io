---
layout: home
title: "subsetselection"
---

# Introduction

We use a gene expression [dataset](references.html/autismdataset) which contains expression levels for more than 50,000 genes collected from autistic and typical subjects. Obviously, this calls for dimensionality reduction techniques. We implement one such technique below. 

### Gene subset selection

We implement the "Nearest Shrunken Centroid Classifier" using the [PAMR](references.html/pamr) package. 

{% highlight r %}
# Nearest shrunken centroid classifier
library(pamr)
# Output labels - 0 represents typical subject, 1 represents autistic subject
output=c(0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)
output=data.frame(output)
library(data.table)
# csv file containing gene expression values of all subjects/samples
data=fread("GeneExpressionInAustisticVsControlSubjects.csv")
genes=data[,1]
data=data[,-1]
unlistdat=unlist(data)
# gene expression values in rows and samples in columns
datmat=matrix(unlistdat,nrow = 54613, ncol = 146)
y=unlist(output)
fulldata=list(x=datmat,y=factor(y))
results=pamr.train(fulldata)
genes[results$nonzero]
{% endhighlight %}

Running the above code returns the following genes(under the column nonzero)

| threshold | nonzero | errors |
|-------|--------|---------|
 0.000|54613|41|
0.152|39994|41|
0.303|30588|40|
0.455|24247|41|
0.606|19380|41|
0.758|15496|41| 
0.909|12325|44|
1.061|9773|44|
1.212|7600|45|
1.364|5861|47|
1.515|4508|46|
1.667|3441|45| 
1.818|2504|45|
1.970|1877|46|
2.121|1388|43|
2.273| 996|43|
2.425| 695|41|
2.576| 470|41|
2.728| 321|40|
2.879| 214|41|
3.031| 126|42|
3.182|  72|41|
3.334|  45|40|
3.485|  25|39|
3.637|  16|38|
3.788|   8|55|
3.940|   6|  59|
4.091|   2 |  64|
4.243|   1  | 64|

Gene subset returned corresponding to the "nonzero" indices:

| ID_REF |
|-------|
 91952_at |
 230739_at |
 221304_at |
 214951_at |
 209973_at |
 206049_at |
 202876_s_at |
 1570303_at|
 1564377_at |
 1561238_at |
 1558949_at |
 1557113_at |
 1555695_a_at |
 1554842_at |
 1554185_at |
 1553618_at |
 1553224_at |
 1552915_at |
 1552712_a_at |
 1552559_a_at |
 1552426_a_at |
 1552343_s_at |
 1552304_at |
 1552275_s_at |
 1552258_at |
 1320_at |
 1294_at |
 1053_at |
 1007_s_at |


