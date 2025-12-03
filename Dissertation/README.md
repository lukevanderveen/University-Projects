# A Study on Flood Detection Using Satellite Imaging and Deep Semantic Segmentation

This Project is an experiment on the works of Zhang et al (2022), which can be found within this repo labeled:

An Interpretable Deep Semantic Segmentation Method for Earth Observation 

It uses the infastructure of the ML4Floods repo, the WORLDFLOODS dataset and a Convultional Neural Network to identify pixel types within multispectural satalite images. The experiemtent looked into firstly implementing this works of Zhang through only his paper and a few recommendations from him on a machine with reduced capabilities (compared to the machine used in the original experiment). Then changing out the driving CNN model from U-net to SegNet to assess for any benifical changes to the process.

To summarise my findings I found that U-Net was better at strictly assigning labels to each pixel, but SegNet was noticably faster at it which is crucial for identifying flooding in terrabytes of satalite imagery.

For a more indepth description of the project and its processes please refer to my Dissertation Report. 