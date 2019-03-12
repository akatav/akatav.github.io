---
layout: home
title: implementation
---
### About nilearn library

_nilearn_ is an open source python module for machine learning and analysis of neuroimaging data. Follow the tutorial [here](https://nilearn.github.io/introduction.html#installing-nilearn){:target="_blank"} to install nipype. 

### Analysis of ABIDE fMRI dataset to predict ADOS scores

Let's try and predict ADOS scores of diagnosed autistic individuals from their fMRI brain scans. That's sure be very an exciting thing to do. 

Let us start with a training set of about 20 individuals who all have autism. In the Phenotypic_V1_0b_preprocessed1.csv, the variable DX_GROUP indicates the presence (or absence) of autism. 0 represents the absence of autism while 1 represents its presence. A full description of the various phenotypic attributes of each individual whose brain was scanned can be found [here](http://fcon_1000.projects.nitrc.org/indi/abide/ABIDE_LEGEND_V1.02.pdf){:target="_blank"}

In nilearn, we use the [fetch_abide_pcp](https://nilearn.github.io/modules/generated/nilearn.datasets.fetch_abide_pcp.html){:target="_blank"} method to download the NIFTI images of the training set. Data will be downloaded to /home/user/nilearn_data/ABIDE_pcp/cpac/nofilt_noglobal

{% highlight python %}
from nilearn import datasets
from nilearn.datasets import fetch_abide_pcp
import pickle
from nilearn.connectome import ConnectivityMeasure
from nilearn.decomposition import CanICA
# download ABIDE dataset for autistic individuals
pitt_abide_ds=fetch_abide_pcp(SUB_ID=[50003, 50004, 50005, 50006, 50007, 50012,50013,50016, 50024, 50025, 50026, 50027], derivatives=['func_preproc', 'rois_cc200'])
funcfiles= pitt_abide_ds.func_preproc

{% endhighlight python %}

NIFTI is a 4d representation of the brain across a predetermined time span. However, for purposes of analysis, the 4d images have to be converted to 2d. We extract the response of each 3d brain cordinate (or voxel) across the given timespan (ie, the blood oxygenation level of each  voxel of the brain varies both in resting and active states across a given time period. So, we get the resulting time vs voxel BOLD activity which is a 2d representation of the 4d .nii image. This process is called masking. 

In fMRI experiments, the BOLD signal recorded in each voxel of the brain aims to record brain activity in resting state or while performing a certain activity. However, in real life, there are many interferences and the BOLD signal recorded will not always purely reflect only brain activity in response to the experimenta stimuli. Because in real time, the brain respond to various other external and internal stimuli such as heartrate, head motion of the scanned individual, breathing etc. So, the recorded signal is often noisy. This is similar to the cocktail party problem in many respects where there are $$n$$ devices that record the conversations of n individuals. Needless to state, each recorder picks up a mixture of the speech signals. The task is to seperate each individual speech signal from the mixture. 

We use the $$CanICA$$ method in nilearn for denoising the images. The resulting 4d image is stored in $$canica.components_img_$$. 

{% highlight python %}
#Perform Canonical ICA on the fmri images to remove noise and seperate independent components
canica = CanICA(n_components=20, smoothing_fwhm=6.,memory='nilearn_cache', mask_strategy='background',  memory_level=2,verbose=10,random_state=0)
canica.fit(funcfiles)
components=canica.components_
# Perform inverse transform on the obtained compoenents 
components_img=canica.masker_.inverse_transform(components)
{% endhighlight %}

This image is then used to extract the time series above using $$NiftiMapsMasker$$ object. 

{% highlight python %}
from nilearn.regions import RegionExtractor
from nilearn import plotting
extractor = RegionExtractor(components_img, threshold=2.,thresholding_strategy='ratio_n_voxels',extractor='local_regions',standardize=True,min_region_size=1350)
# Just call fit() to process for regions extraction
extractor.fit()
# Extracted regions are stored in regions_img_
regions_extracted_img = extractor.regions_img_
# Total number of regions extracted
n_regions_extracted = regions_extracted_img.shape[-1]
# Visualization of region extraction results
title = ('%d regions are extracted from %d components.' % (n_regions_extracted, 20))
plotting.plot_prob_atlas(regions_extracted_img,view_type='filled_contours',title=title, threshold=0.008)
{% endhighlight %}

Our task is to predict ADOS scores of test samples based on the time series and ADOS score information of the training samples. The assumption is that depending on the connectivity between the various regions of the brain of an individual, the prevalence of autism (and values of ADOS scores) can be predicted. Here, we use the rois_cc200 atlas to extract 200 regions of interest in the brain. 

nilearn has the $$ConnectivityMeasure$$ method readily available to get the connectivity matrices for the training samples. The obtained connectivity matrix is simply the correlation between the time series of the various brain regions. Once, all the connectivity matrices are obtained, we use these matrices as the new training set. Since these matrices are symmetric matrices, it is necessary to extract the upper or lower triangular portions of these matrices. 

{% highlight python %}
from nilearn.connectome import ConnectivityMeasure
conn_est = ConnectivityMeasure(kind='partial correlation')
conn_matrices = conn_est.fit_transform(pitt_abide_ds.rois_cc200)

from nilearn.connectome import sym_to_vec
X = sym_to_vec(conn_matrices)
{% endhighlight %}

The next step is to apply suitable learning algorithms that accept the connectivity matrices as training set alongwith the ADOS scores for these individuals as the training target values  to predict ADOS scores for unseen brain images. We can also setup a cross validation routine to accomplish the same. 

We use Support Vector Regression using GridSearch. 

{% highlight python %}
pipe_svr = Pipeline([('scl', StandardScaler()),('reg', MultiOutputRegressor(SVR()))])
grid_param_svr = {
    'reg__estimator__kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
    'reg__estimator__gamma':[0.1,1.0,10],
    'reg__estimator__C':[1e0, 1e1, 1e2, 1e3],
    'reg__estimator__gamma': np.logspace(-2, 2, 5)
}
gs_svr = (GridSearchCV(estimator=pipe_svr,param_grid=grid_param_svr,cv=2,scoring = 'mean_squared_error',n_jobs = -1))
gs_svr = gs_svr.fit(X[:8],k[:8])
ypred=gs_svr.best_estimator_.predict(X[9:12])
print(mean_squared_error(ypred,y[9:12]))
{% endhighlight %}

