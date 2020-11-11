# Readme
# Data Files
The data is in the format of h5 files. There are files containing  the x,y positions, x,y velocities and the orientation on each row for each iteration per environmental setup, they are named "basic_xy.h5", "mod25_xy.h5" etc. There are also h5 files containing the visual embeddings on each row for each iteration named "basic_embeddings.h5" etc. These data files are stored on google drive at: https://drive.google.com/drive/folders/1u6je2vQ8AY9Snm7UjnZKuLLfpUgFWSxe?usp=sharing

Using these data files along with the saved hippocampal in the "trained_models" folder model it is quick and easy to generate the activity of the hippocampal model using model.predict(). Due to the speed of this operation and the fact I was iterating the models a lot it did not make sense to save the activities for a given a model and so the code is set up to use predict() in the rate_maps.py script.
## Code

There are 8 files in the repo:
denoiser.py -  building and training the models.
rate_maps.py - generating hippocampus activity and building rate maps and their plots
analyse_dataframes.py - use rate maps to build dataframes with statistics, generate the other plots
The rest shouldnt be necessary and are pretty self explanatory.

## Requirements

There is a requiremnets.txt file that contains the conda instruction to build the environment.  


