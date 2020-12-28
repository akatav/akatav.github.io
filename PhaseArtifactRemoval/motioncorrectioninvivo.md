---
layout: home
title: "motioncorrectioninvivo"
---

### Correcting for phase artifacts due to bidirectional scanning in imaging

We detail how to remove phase artifacts in images obtained from in-vivo imaging. We use the [NoRMCorre](../references.html/NoRMCorre) library to study the same. We also consulted some papers in this regard such as [this](../references.html/phaseartifactremoval) one.  

1. Let _Y_ be the matrix with dimensions _(x,y,n)_ representing an image with _n_ frames (chosen from _N_ frames of the original image (_n_ &lt;&lt;&lt; _N_). Each frame has dimensions _(x,y)_.  
2. Get mean of _n_ frames resulting in the resultant mean image _M_ of dimension _(x,y)_.  
3. Get _min(M)_  
4. Get odd and even rows of the mean matrix _M_ as two matrices, say, _M_<sub>o</sub> and _M_<sub>e</sub>  
5. Upsample _M_<sub>o</sub> and _M_<sub>e</sub> by some factor _usfac_ using [_imresize_](../references.html/imresize) function. Let the upsamples matrices be _M_<sub>uo</sub> and _M_<sub>ue</sub>.  
6. Get fourier transforms of _M_<sub>uo</sub> and _M_<sub>ue</sub> using [_fft_](../references.html/fft). Let these matrices be _F_<sub>1</sub> and _F_<sub>2</sub>.  
7. Get product _P_ of _F_<sub>1</sub> and conjugate of _F_<sub>2</sub>.  
8. Get inverse fourier transform, that is, [ifft](../references.html/ifft) of _P_. Let it be _C_.
9. Get absolute value of _C_. Let it be _abs_.  
10. Get mean of _abs_ along the upsampled dimension, here _y_. This returns _usfac*y_ mean values. Get max of these values and get it's index _ind_.  
11. Reverse zero frequency shift _I_ using [_ifftshift_](../references.html/ifftshift) along the upsampled dimension _usfac*y_.  
12. Return shift due to phase artifact as _I[ind]/usfac_.  
