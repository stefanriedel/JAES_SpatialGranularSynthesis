# JAES_SpatialGranularSynthesis
This repository hosts python code and data of the publication 'Perceptual Evaluation of Envelopment and Engulfment using Spatial Granular Synthesis'.

Dependencies:
* numpy
* scipy
* matplotlib
* joblib

A spatial granular synthesis technique was used to investigate the effect of the temporal and directional density of sound events on listener envelopment and engulfment. The idea behind spatial granular synthesis can be seen in the following sketch:

<img src="/Figures/SGS/SGS_sketch.PNG" alt="drawing" width="800"/>

Small audio segments are sampled from a single-channel audio buffer and distributed in space. In the experiment we used a multichannel loudspeaker system composed of three height layers. Pre-rendered plots can be found in the /Figures subdirectory, and to render binaural audio use the *renderBinauralEvaluationStimuli.py* script. Some pre-rendered audio files are found at \BinauralAudio. These files are used by the  *plotBinauralEvaluation.py* script.
