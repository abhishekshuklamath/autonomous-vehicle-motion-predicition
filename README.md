# autonomous-vehicle-motion-predicition

This is a subset of the Interpret challenge http://challenge.interaction-dataset.com/prediction-challenge/intro
The aim is to predict the motion of the ego vehicle given its motion for the past 10 seconds. The training data also gives the information about neighbours of the ego vehicle which may themselves be cars or pedestrians. 

A first step was to process the data to create features. The first attempt I took was to not worry about the neighbours' role in the driving prediction process.  I converted the temporal coordinates of the ego vehicle into features. Since we had positions, that is the (x,y) coordinate, at 10 time stamps in the past it gave rise to 10*2=20 features of the agent.
Instead of building an autoregressive model I decided to build 20 independent models to predict 20 coordinates for the next 3 seconds of movement. 
Then I trained a linear regression model and got rsme validation error of 0.3425. 
This is pretty good considering we did not use any deep learning models.

After the linear model, I decided to train a neural net model with 50 hidden layers using the code from the class. This was a pretty good model and gave a validation error of . Then I decided to use this model for the test data too and got the rsme as 0.36959. Considering I did not use any ``other" roles for prediction I decided to include the neighbour data to see if prediction could be improved.


After the neural net with no neighbour data I tried to incorporate the neighbour data by choosing the closest neighbour(other than the ego itself) to the ego vehicle and including their position as the features in addition to the position of the ego vehicle.  This model required a lot of preprocessing because we need to find the vehicle closest to the ego vehicle in all the training set. After preprocessing and training I found the rsme on the validation data set to be  0.343, which gave a test score of 0.35136 with the Linear Regression model. With the neural net model incorporating neighbour data I got a validation rsme of 0.4901.

These results suggest that linear regression is a pretty good model for making vehicle position predictions.
