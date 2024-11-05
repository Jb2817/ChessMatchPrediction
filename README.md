# ChessMatchPrediction
Chess Match Prediction (CMP) is a python program that trains machine learning models on a user's pgn files and picks the superior one to predict the result of a player's game against a new opponent. Fetaure generation is employed to better model performace. After a prediciton is made the user can include the results of the match which is used to retrain the model employed. Allowing further improvement on the performance of the model. Interested psrticipants can look at the coefficients for select models to see which features had the most impact on prediciton -  a potenial hint towards user's in game improvement.

#How to run 
1. Collect pgn files (I reccomend opening tree: [https://www.openingtree.com/ ](https://www.openingtree.com/))
2. Place all programs and pgn files into a folder and open a terminal window from that folder
3. Enter: python3 estimator.py (Your first pgn).pgn (Optional additional pgns ).pgn
4. Follow script live from terminal

# How it works
CMP first processes the pgn files:
  - Parses the pgn file through series of regex expressions and function calls
  -  Iterates through individual matches to converts each game into a record;recording information like (game outcome, time control, opening used, players elo, etc)
  - All records appended into a pandas dataframe

CMP Data Cleansing
  - Removes redudant information
  - Type conversions
  - Standardizing openings ex: Sicillian Defense: Najdorf Variation -> Sicillian Defense

CMP Feature Generation
  - Win/Draw/Loss rates
  - Performance indicator
  - Openings conversion

CMP Trains Models
  - 7 models are used -  all are trained and tested on a 80/20 dataset split
  - Evaluations of models:
    Since CMP is a multilabel classification problem Sklearn accuracy_score is employed
    The model with the highest accuracy score is chosen

CMP calls main.py
  - Through a series of questions and answers CMP records the same information from the user that would be seen in a record of the testing dataset. With this the model makes a prediction and uses the response to train itself again. 



    
