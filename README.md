# spaceship-titanic
A fun dataset to practice some classifiers.

## Dataset
The dataset was acquired from [kaggle](https://www.kaggle.com/competitions/spaceship-titanic/data). The training dataset had 8693 records and 13 features.

The dataset contains the below attributes:-

+ PassengerId - A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group. People in a group are often family members, but not always.
+ HomePlanet - The planet the passenger departed from, typically their planet of permanent residence.
+ CryoSleep - Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.
+ Cabin - The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.
+ Destination - The planet the passenger will be debarking to.
+ Age - The age of the passenger.
+ VIP - Whether the passenger has paid for special VIP service during the voyage.
+ RoomService, FoodCourt, ShoppingMall, Spa, VRDeck - Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.
+ Name - The first and last names of the passenger.
+ Transported - Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.

## Data Preprocessing
I had a lot of fun doing data preprocessing on this dataset. Some of the techniques I used are mentioned below:-
+ I contemplated deriving the group from the PassengerId, and using that in the model. I did not find any relation between group and the label, so I ended up dropping that column.
+ I derived three difference columns from Cabin- deck, num and side
+ Buckeked Age into groups like 10s, 20s, 30s etc.
+ Label encoded all the categorical columns 

## Models
I built various classifications models to compare the performance and efficiency - Naives Bayes, XGBoost, Random Forests. The confusion matrix and accuracy can be found in the notebook.
For the first time, I also used the voting classifier to compare the various classifiers like Logistic Regression, Support Vector Machine, Decision Tree Classifier, XGBoost and Naive Bayes. I also ran a GridSearchCV for hyperparamter tuning of the RandomForest. The best performance in kaggle was found to be the Random Forest with parameters determined by GridSearchCV. This model just proved how effective GridSearchCV is for identifying the optimal parameter values for a classifier.


