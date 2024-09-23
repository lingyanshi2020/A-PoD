# A-PoD
reference article: https://www.nature.com/articles/s41592-023-01779-1

A-PoD is a deconvolution program to convert a diffraction limited microscopy image to a super-resolved image. This code is written in Python 3.8.7c1 with following plugins. 



Tensorflow-gpu 2.6.0
Numpy 1.20.3
Pillow
Sci-kit image
opencv
pandas

Computer specification: 
1. Xeon W-2145 CPU, 64 GB RAM, and NVIDIA Quadro P4000 GPU
2. Intel Core i7-9700 CPU, 16 GB RAM, and NVIDIA GeForce GTX 1660 Ti
3. Intel Xeon Gold 5317 CPU, 128 GB RAM, and NVIDIA RTX A4500


# How to run
Python and the plugins should be installed. After running the code (A-PoD_GUI.py or A-PoD_GUI_Single_PSF.py), the files of PSF and image should be loaded.

For demonstration of the program, A-PoD_GUI_Single_PSF.py should be run. (Image file: HEKCell_4x.tif, PSF file: PSF_HEK_4x.tif)
The parameters can be adjusted depending on computational conditions. For example, the number of maximum address can be increased, but it needs more calculation time. 
If memory size is not enough, the number of spot can be reduced. "number of spot" is maximum number of spots that are calculated in a single step.
Learning rate is a parameter for Adam solver. This parameter is proportional constant to calculate the real learning rate based on PSF size. (ex. Learning rate = 0.75, then real learning rate = 0.75*PSF_size)
GPU memory size is the maximum GPU memory. If GPU has 8GB RAM, then the prarameter can be 8192. Then, automatically, the program utilize 90% of maximum RAM (7373 MB).
The last parameter is reblurring parameter. After deconvolution, to avoid overfitting, Gaussian filtering will be done. The parameter means sigma of the Gaussian filter. To preserve raw deconvolution results, we can set the parameter as 0.

To deconvolve SRS images, we need two PSFs (pump beam and Stokes beam). For the purpose, A-PoD_GUI can open two PSFs. 

The result will be exported in the folder, "/deconvolved".

# parameter for the test image (08/25/2023)

number of maximum address: 1000
Number of spot: 5000000
Learning rate: 0.5
Reblurring: 2

# Update (08/25/2023)
A function to exclude low frequency background signal is added. Due to the change, time to process is decreased. 

# Update (09/14/2023)
Gradient function is simplified. 

# Update (12/14/2023)
Gradient function and background thresholding method are changed. 

# Update (09/22/2024)
Raw image files were uploaded. (Retina tissue and two beads images) 

If you have any qustions or suggestions, please send email to the address below.

hojang@ucsd.edu

