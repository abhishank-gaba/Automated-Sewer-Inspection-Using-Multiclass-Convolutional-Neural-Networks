# Automated-Sewer-Inspection-Using-Multiclass-Convolutional-Neural-Networks

This is a research paper that I, along with one other co-author, worked on over the past year (September, 2018 - Currently). 
My co-author and I are trying to get this paper published, thus actively iterating on the paper's implementation of detecting defects in sewer pipes. 


Brief Description:

A deep multi-class convolutional neural network is used to detect critical points in pipes;
specifically joints, connections, and manholes. An accuracy of 99.7% is achieved using a dataset
composed of 7 pipe videos of 2 concrete storm, 1 PVC storm, 2 concrete sanitary and 2 PVC
sanitary sewer pipes. Three hyperparameters are varied: learning rate, batch size, and class
weights assess their impact on training. In sum, the high accuracy implies that the model
may be overfitted to the given dataset, despite the data augmentations, 5 fold validations, and
dropouts used.

