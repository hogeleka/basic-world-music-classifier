# basic-world-music-classifier
Simple app which uses machine learning to classify audio under Latin American/Caribbean, Sub-Saharan Africa, South Asia, and South East Asia. Built off of [the work of _Parul Pandey_] (https://towardsdatascience.com/music-genre-classification-with-python-c714d032f0d8) 


### Setup
* Clone the repository
* Open the project in Pycharm. This is helpful so you can know and download all the packages you need (we used`numba` package version 0.45.0 not most recent version of  `numba`)
* In your project interpreter, make sure all the required imported packages are installed into your python environment. We recommend Python 3.6 or 3.7 as some packages do not work with python 3.8 and above. 

### Running the app
* Navigate to `app.py` and run the `main` function
* Open up `localhost:5000/` in your browser
* Choose a wav file you would like to predict, and click `Submit`
* Results are displayed (see explanation in next section for what "Type 1" and "Type 2" mean)

### Building your own model
* Create a directory named `wm_regions`, and another directory named `csv`
* Navigate to `wm_regions`, and create four sub-directories named
`latin_america_carrib`, `south_asia`, `south_east_asia`, and `sub_sahara_africa`
* In each sub-folder in `wm_regions`, download as many (we used 50-70 per region) 1 minute samples (to be sure the audio file is long as the script needs it to be, audio should be longer than 60s) of `.wav` audio. These would be used to train your model
* Navigate to `src/CSVGenerator.py` and run that script. It will output a csv file in the `csv` directory which you will use to create your model
* Navigate to `src/ModelGenerator.py`. At bottom of that file, set the variable `modelFileName` to be the file path of whatever you would like to save your model as. __It must be a .h5 file.__  Feel free to experiment with number of epochs used to train the model. Run this script; this will use the data written in the csv file to create a machine learning model. 
* To test that your model works, navigate to `src/MusicRegionPredictor.py`. Change the variable `modelName` at the top of the file to be the path of the .h5 model you generated in step above. 
At the bottom, call the function `predictSong`, making sure to pass in the path to a .wav audio file (must be longer than 1 minute) as the parameter. It should output two lists, each list is a ranking of the model's prediction for the regions. From most similar (first) to least similar (last).
The difference between the two lists is this: the __Type 1__ list accumulates the relative scores on a minute-by-minute basis per region, and aggregates it. The __Type 2__ list just uses the top predicted region for each one minute chunk and aggregates them. 
* In `src/ModelCrossValidator.py`, run that script to perform some basic cross-validation



### Current limitations/Possible improvements
* Limited to 4 regions
* We used about 50-70 1 minute chunks per region to train our model. We used about 9-12 songs per region. So more variety would be good
* Improvements needed on our use of Keras/tensorflow 
* UI improvements

