from tqdm import tqdm
import chess.pgn
import re
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from warnings import filterwarnings
filterwarnings('ignore')

import sys

def parse_match(game):
    '''
    Parses an individual match within pgn file - records all information regarding the game
    '''
    pattern = r'\[(\w+) "([^"]+)"\]'
    user = username
    matches = re.findall(pattern, game)

    chess_info = {key: value for key, value in matches}

    return chess_info

def parse_pgn(pgn, matches):
    '''
    Reads the pgn file and processes every game 
    '''
    while True:
        game = chess.pgn.read_game(pgn)
        if game is None:
            break
        matches.append((parse_match(str(game))))

    games_df = pd.DataFrame(matches)
    return games_df

def streak(df, val, streak_column):
    '''
    Feature for recording winning/losing/drawing streak a player currently has
    '''
    recordings = []
    count = 0
    for index, value in df['Result'].items():
        if value == val:
            count += 1
        else:
            count = 0
        recordings.append(count)

    df[streak_column] = pd.Series(recordings).shift(1, fill_value=0)
    return df[streak_column]

def openings_rating(df, mapping):
    '''
    Gives each opening played an individual score based on how many wins associated 
    
    Openings play a larger role for non-amateur players (>2100), and less for amateurs
    Opening_rating can lead models to drawing irrelevant conclusions and should be used 
    primarily by players with higher ratings OR a large(>1000) game history
    '''
    recordings = []
    count = 0
    for index, value in df['Result'].items():
        if value == val:
            count += 1
        else:
            count = 0
        recordings.append(count)

    df[streak_column] = pd.Series(recordings).shift(1, fill_value=0)
    return df[streak_column]

def construct_models(storage, X_train, X_test, y_train, y_test):
    from sklearn.naive_bayes import GaussianNB
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    storage.append((nb, accuracy))
    print("Naive Bayes Accuracy:", accuracy)

    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    storage.append((lr, accuracy))
    print("Linear Regression Accuracy:", accuracy)

    from sklearn.svm import SVC
    svm = SVC(decision_function_shape='ovo')
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    storage.append((svm, accuracy))
    print("SVM Accuracy:", accuracy)

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    storage.append((rf, accuracy))
    print("Random Forest Accuracy:", accuracy)

    from xgboost import XGBClassifier
    xgb = XGBClassifier()
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    storage.append((xgb, accuracy))
    print("XGBoost Accuracy:", accuracy)

    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    storage.append((knn, accuracy))
    print("KNN Accuracy:", accuracy)

    from sklearn.neural_network import MLPClassifier
    nn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
    nn.fit(X_train, y_train)
    y_pred = nn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    storage.append((nn, accuracy))
    print("Neural Network Accuracy:", accuracy)

    storage.sort(key=lambda x: x[1], reverse=True)
    return storage, storage[0]

def analyze_lr_model(model, feature_names):
    """
    Analyzes the linear regression model coefficients and creates report
    of feature importance.
    """

    coefficients = model.coef_
    intercepts = model.intercept_
    
    coef_df = pd.DataFrame(
        coefficients,
        columns=feature_names,
        index=['Loss', 'Draw', 'Win']
    )
    
    coef_df['Intercept'] = intercepts
    importance_df = pd.DataFrame()
    importance_df['Feature'] = feature_names
    importance_df['Average Absolute Impact'] = np.abs(coefficients).mean(axis=0)
    importance_df = importance_df.sort_values('Average Absolute Impact', ascending=False)
    
    print("\nMODEL ANALYSIS")
    print("=" * 50)
    
    print("\nTop 5 Most Important Features:")
    print("-" * 50)
    for idx, row in importance_df.head(5).iterrows():
        print(f"{row['Feature']:<20} Impact: {row['Average Absolute Impact']:.4f}")
    
    print("\nDetailed Coefficient Analysis:")
    print("-" * 50)
    print("\nPositive values indicate increased likelihood of that outcome")
    print("Negative values indicate decreased likelihood of that outcome\n")
    
    for outcome in ['Loss', 'Draw', 'Win']:
        print(f"\n{outcome} Outcome - Top 5 Contributing Features:")
        print("-" * 40)
        outcome_coeffs = coef_df.loc[outcome].sort_values(ascending=False)
        for feat, coef in outcome_coeffs.head().items():
            print(f"{feat:<20} Coefficient: {coef:>8.4f}")
            
    return coef_df, importance_df


username = input("Please enter your username as it appears in your pgn ").strip()
pgns = []
for file_path in sys.argv[1:]:
    with open(file_path) as f:
        parsed_content = parse_pgn(f, [])
        pgns.append(parsed_content)

user_data = pd.concat(pgns, ignore_index=True)

classical_win_rate = user_data["Result"].value_counts().get("1-0", 0) / len(user_data["Result"])
classical_draw_rate = user_data["Result"].value_counts().get("1/2-1/2", 0) / len(user_data["Result"])
user_data["WinRate"] = classical_win_rate
user_data["DrawRate"] = classical_draw_rate

user_data.sort_values(by=['Date'])

X = user_data.drop(['EndDate', 'EndTime', 'Link', 'Termination', 'UTCTime', 'UTCDate', "ECOUrl", 'Event', 'Site', "Round", 'Result', 'CurrentPosition', "Timezone"], axis=1)
X["Date"] = X["Date"].apply(lambda x: datetime.strptime(x, "%Y.%m.%d").toordinal())
X["StartTime"] = X["StartTime"].apply(lambda x: datetime.strptime(x, "%H:%M:%S")).apply(lambda x: x.hour)

y = user_data['Result']
val, openings = 0, {}
for opening in X['ECO'].unique():  
    if opening not in openings:
        openings[opening] = val
        val += 1
X["opening"] = X['ECO'].map(openings)
X.drop('ECO', axis=1, inplace=True)
X["White"] = 1
X["Black"] = 0
y = y.map({"1-0": 2, "0-1": 0, "1/2-1/2": 1})


X = X[y.notna()]
y = y[y.notna()]
y = y.astype(int)

X['WhiteElo'] = pd.to_numeric(X['WhiteElo'], errors='coerce')
X['BlackElo'] = pd.to_numeric(X['BlackElo'], errors='coerce')
X['TimeControl'] = X['TimeControl'].str.extract(r'(\d+)').astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

models = []
m, model = construct_models(models, X_train, X_test, y_train, y_test)
print("Completed")
X_train = X_train.sort_values(by=['Date']).reset_index(drop=True)
latest_win_rate = X_train["WinRate"].iloc[-1]
latest_draw_rate = X_train["DrawRate"].iloc[-1]


from main import main

main(m, X_train, y_train,latest_win_rate, latest_draw_rate)

