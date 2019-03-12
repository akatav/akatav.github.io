---
layout: home
title: nifti
---

### The NIFTI file format

NIFTI (Neuroimaging Informatics Technology Initiative) is a new format for storing brain images. A single fMRI session generates several thousand images of the brain. Older formats such as the DICOM stored individual DICOM images with their own exclusive headers for all these thousands of images. However, most of these thousands of headers were identical which makes storage redundant. NIFTI format combines all the DICOM images without the headers into a single file. As a result, in the case of fMRI where the brain is imaged in all three dimensions (x,y,z) across different time points, the entire 4-dimensional image (3 dimensions, $$4^{th}$$ dimension as time) is stored in a single file. This elimininates the need for storing individual headers and frees up space. 

