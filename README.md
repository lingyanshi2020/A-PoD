# A-PoD

A-PoD is a deconvolution program to convert a diffraction limited microscopy image to a super-resolved image. This code is written in Python 3.6.0 with following plugins. 


Tensorflow-gpu 1.15

Numpy 1.17.3

Pillow

Sci-kit image

opencv

pandas



# How to run
First of all, python and the plugins should be installed. In the JupyterNotebook file, the details of this code is explained.
In order to make a test with the code, the file, A-PoD_2D.ipynb file should be opened. The important parameters are the psf and filename. The two example files of a PSF and an image are included in this account. We can simply type in the foler name including the file name. 

The spatial frequencies of PSF and image should be matched. The PSF file in this example was generated using PSF generator plugin in ImageJ. Considering the experimental conditions, we can precisely define the PSFs.

Additional parameters
1) vir_num_scaling_factor: parameter to increase or decrease the number of total virtual emitters.
2) background_threshold: normally, it is 0. But, if we need to adjust offset value for the image, we can adjust this parameter.
3) numbmaxaddr: maximum number of emitters that this program handle in a single calculation. Based on this number, the group of every virtual emitter is separated to several small groups. This number is the high limit of the number of each small group.
4) signal_bg_ratio: This ratio represent the intensity of each single virtual emitter. If this number is 1, the sum of total virtual emitter intensity is same with total intensity of the image. But, considering the noise, offset, and other effects, the number is set below 1, around 0.8~0.9.
5) iter_criterion: this number is related to the convergence of Adam solver. Calculation with low iter_criterion generates more precise result, but the calculation needs more time.


