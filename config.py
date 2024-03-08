DATAPATH = "datasets/"
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

TARGET='Transported'

FEATURES_DROP = ['PassengerId', 'Name', 'Cabin', 'Age']

FEATURES_TO_ENCODE = ['HomePlanet', 'Destination', 'deck', 'num', 'side', 'AgeGroup']

FEATURES_NUMERICAL = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

FEATURES_BOOL = ["CryoSleep","VIP"] #Boolean features which need to be converted to 0 (False) and 1 (True)