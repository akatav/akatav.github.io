---
layout: home
title: "eeganalysis"
---
{% highlight python%}

## READ A SAMPLE FILE TO EXTRACT SAMPLING FREQUENCY
import mne
import mne_features
mcifile=mne.io.read_raw_eeglab("/examples/ClassifierTestsample/aMCI/CAUH_298.set")
{% endhighlight %}
