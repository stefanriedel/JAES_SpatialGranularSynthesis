# JAES_SpatialGranularSynthesis
This repository hosts python code and data of the publication 'Perceptual Evaluation of Envelopment and Engulfment using Spatial Granular Synthesis'.

Dependencies:

* numpy

* scipy

* matplotlib

* joblib

In the contribution a spatial granular synthesis technique was used to investigate the effect of the temporal and directional density of sound events on listener envelopment and engulfment. The idea behind spatial granular synthesis can be seen in the following sketch:

<img src="/Figures/SGS/SGS_sketch.PNG" alt="drawing" width="800"/>

Small audio segments are sampled from a single-channel audio buffer, then distributed in space. In the experiment we used a multichannel loudspeaker system with multiple height layers.
