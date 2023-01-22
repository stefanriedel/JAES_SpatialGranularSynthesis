# ActaAcustica_SpatialGranularSynthesis
This repository hosts python code and data of the publication 'Perceptual Evaluation of Envelopment and Engulfment using Spatial Granular Synthesis'.

Dependencies:
* numpy
* scipy
* matplotlib
* soundfile
* joblib

A spatial granular synthesis technique was used to investigate the effect of the temporal and directional density of sound events on listener envelopment ('being surrounded by sound') and engulfment ('being covered by sound'). The idea behind spatial granular synthesis can be seen in the following sketch:

<img src="/Figures/SGS/SGS_sketch.PNG" alt="drawing" width="800"/>

Audio 'grains' of length $L$ seconds are seeded from a single-channel buffer $x(t)$, where the random seed positions can be limited to a region of $Q$ seconds in the buffer. Grains are spatially encoded to a multichannel output, e.g. every $\Delta t$ seconds. The approach is equivalent to a time-varying multichannel FIR filter. In the experiment we used a multichannel loudspeaker system composed of three height layers for the reproduction of the synthesized output. The plots of the paper are reproducible and rendered to the /Figures subdirectory. 
