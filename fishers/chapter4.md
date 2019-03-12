---
layout: home
title: "timeline"
---

# Mathematical derivation of Fisher's discriminant(s)

Let $$\mu_g$$ be the mean of _X_. Let the mean of each of the _K_ groups of data (denoted by $$X_k$$) be $$\mu_k$$, $$1 \le k \le K$$. We aim to find the Fisher's discriminant _w_ such that $$Y=w.X$$. This was explained in the previous chapter. 

Let $$X'$$ be the set of orthogonal projections of $$X$$ on the discriminant _Y_. 

$$
\begin{equation}
\mu_g' = w.\mu_g\tag{1}\label{eq:one}
\end{equation}
$$

be the orthogonal projection of $$\mu_g$$ on _Y_. 

Similarly, 

$$
\begin{equation}
\mu_k'=w.\mu_k\tag{2}\label{eq:two}
\end{equation}
$$ 

be the orthogonal projections of each $$\mu_k$$ on _Y_. 

As described in the previous chapter, _FDA_ tries to find _w_ by maximizing the distance between the group means while minimizing the distance between points within a group _k_ and their respective means in the projected space $$d'$$. 

Let $$S_w'$$ be the __within-class covariance__ and $$S_b'$$ be the __between-class covariance__ in the projected space. 

$$
\begin{equation}
S_w' = \sum_{k=1}^K (\mu_k'-X_k').(\mu_k'-X_k')^T\tag{3}\label{eq:three}
\end{equation}
$$

$$
\begin{equation}
S_b' = \sum_{k=1}^K (\mu_k'-\mu_g').(\mu_k'-\mu_g')^T\tag{4}\label{eq:four}
\end{equation}
$$

Note that, $$S_w'$$ and $$S_b'$$ represent the covariances within and between the various classes/groups of the data.

We know that covariance of _X_ is: 

$$
E[X] = E[(X-E(X))(X-E(X))^T]\tag{5}\label{eq:five}
$$

Covariance of $$Y$$ is:

$$E[w.X] = E[(wX-E(wX))(wX-E(wX))^T]\tag{6}\label{eq:six}$$

$$E[wX] = E[w(X-E(X))(X-E(X))^Tw^T]$$

$$E[wX] = w.E[(X-E(X))(X-E(X))^T].w^T$$

Substituting equation (5), 

$$E[wX] = w.E[X].w^T\tag{7}\label{eq:seven}$$

From equation (7), we can rewrite $$S_w'$$ and $$S_b'$$ in terms of $$S_w$$ and $$S_b$$ as follows:

$$S_w'=w.S_w.w^T\tag{8}\label{eq:eight}$$

$$S_b'=w.S_b.w^T\tag{9}\label{eq:nine}$$

As mentioned, we try to maximize $$S_b'$$, which is the distances between all the $$\mu_k'$$ of each of the _K_ classes. It turns out that this is the same as maximizing the distance from each $$\mu_k'$$ and the global mean $$\mu_g'$$. 

The optimization problem is as follows: 

$$J(w) = \frac{S_b'}{S_w'} $$

$$J(w) = \frac{wS_bw^T}{wS_ww^T} $$

Rewriting the above equation as a Lagrangian, we get: 

$$J(w) = wS_bw^T + \lambda (wS_ww^T-1)$$

Differentiating w.r.t _w_ and setting to 0, 

$$\frac{\partial J}{\partial w} = 2wS_b +2 \lambda wS_w $$

$$wS_b + \lambda wS_w = 0$$

$$wS_b = -\lambda wS_w\tag{10}\label{eq:ten}$$

_w_ is a $$[d,1]$$ vector, $$S_b$$ and $$S_w$$ are $$[d,d]$$ matrices. $$\lambda$$ is a scalar. The above can be rewritten as follows in the form of the __Generalized Eigenvalue problem__

Multiply both sides by $$S_w^{-1}$$, 

$$wS_w^{-1}S_b = -\lambda w\tag{11}\label{eq:eleven}$$

$$w=eigen(S_w^{-1}S_b)\tag{12}\label{eq:twleve}$$ gives value of the Fisher's discriminant _w_.  We are interested in the first $$min(K-1, d)$$ eigen vectors in descending magnitude (of eigen values). 

Multiplying the original data _X_ by _w_ as obtained above gives the orthogonal projection of _X_. 

### Differences between PCA and Fishers

We see that Fishers discriminants are found by eigen decomposition of $$S_w^{-1}S_b$$. In contrast, PCA find its components by the eigen decomposition of the covariance matrix of _X_, $$eigen(S_t)$$ where $$S_t$$ is the covariance of the entire data, _X_. 

We note here that $$S_t=S_w+S_b$$. We will derive this also for the sake of thoroughness. 

Thus, we see that PCA and _FDA_ are different in the covariance matrices they decompose. Also, from this, we understand that _PCA_ decomposes the covariance matrix devoid of any class labels unlike _FDA_. 

### Total covariance is the sum of within-class and between-class covariance

$$S_t = S_w + S_b$$

$$S_w = \sum_{k=1}^{K} \sum_{i=1}^{n} (x_i^k-\mu_k)(x_i^k-\mu_k)^T\tag{13}\label{eq:twelve}$$

$$S_b = \sum_{k=1}^{K} (\mu_k-\mu_g)(\mu_k-\mu_g)^T\tag{14}\label{eq:thirteen}$$

$$S_t = \sum_{k=1}^K \sum_{i=1}^{n} (x_i^k - \mu_g)(x_i^k - \mu_g)^T\tag{15}\label{eq:fourteen}$$

$$S_t$$ can be rewritten as below:

$$S_t = \sum_{k=1}^K \sum_{i=1}^{n} (x_i^k-\mu_k+\mu_k-\mu_g)(x_i^k-\mu_k+\mu_k-\mu_g)^T$$

$$S_t = \sum_{k=1}^K \sum_{i=1}^{n} (x_i^k - \mu_k)(x_i^k-\mu_k)^T + (\mu_k-\mu_g)(\mu_k-\mu_g)^T
\\ + (x_i^k - \mu_k)(\mu_k-\mu_g)^T + (\mu_k-\mu_g)(x_i^k - \mu_k)^T$$

Substitute (14) and (15), we get: 

$$S_t = S_w + S_b + \sum_{k=1}^K \sum_{i=1}^{n} (x_i^k - \mu_k)(\mu_k-\mu_g)^T 
\\	+ (\mu_k-\mu_g)(x_i^k - \mu_k)^T\tag{15}$$

Note that, 

$$\sum_{k=1}^K \sum_{i=1}^{n} (x_i^k - \mu_k) = 0$$. 

This is because, sum of the differences of the samples around respective mean is always 0. 

Substituting in (15), we get:

$$S_t = S_w + S_b$$

<a class="continue" href="chapter5.html">Next chapter</a>

