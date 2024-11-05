import pandas as pd
from datetime import datetime
from sklearn.linear_model import LogisticRegression
import numpy as np

def analyze_lr_model(model, feature_names):
    """
    Analyzes the linear regression model coefficients and creates a detailed report
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
    
    print("\nMODEL ANALYSIS REPORT")
    print("=" * 50)
    
    print("\nTop 3 Most Important Features:")
    print("-" * 50)
    for idx, row in importance_df.head(3).iterrows():
        print(f"{row['Feature']:<20} Impact: {row['Average Absolute Impact']:.2f}")
            
    return coef_df, importance_df

def collect_user_input(latest_win_rate, latest_draw_rate, X_train):
    """
    Collects information about a chess match from the user to predict the outcome.
    Returns a DataFrame with the same format as X_train.
    """
    data = {}
    
    now = datetime.now()
    data["Date"] = now.toordinal()
    data["StartTime"] = now.hour
    data["WhiteElo"] = int(input("Enter the White Elo rating: "))
    data["BlackElo"] = int(input("Enter the Black Elo rating: "))
    data["TimeControl"] = float(input("Enter the time control (seconds): "))
    opening_input = input("Enter the ECO opening ID (unique integer), or press Enter to skip: ").strip()
    data["opening"] = int(opening_input) if opening_input else -1
    data["White"] = int(input("Is the user playing White? Enter 1 for Yes, 0 for No: "))
    data["Black"] = 1 - data["White"]
    data["WinRate"] = latest_win_rate
    data["DrawRate"] = latest_draw_rate


    user_input = pd.DataFrame([data])
    missing_cols = set(X_train.columns) - set(user_input.columns)
    for col in missing_cols:
        user_input[col] = 0  

    user_input = user_input.reindex(columns=X_train.columns)
    
    return user_input

def predict_match(model, user_input):
    """
    Predicts the outcome of a match given a model and user input.
    Returns the predicted outcome label.
    """
    try:
        prediction = model.predict(user_input)[0]
        outcome_map = {2: "Win", 0: "Loss", 1: "Draw"}
        return outcome_map.get(prediction, "Unknown outcome")
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        print("Model features:", model.feature_names_in_ if hasattr(model, 'feature_names_in_') else "Unknown")
        print("Input features:", user_input.columns.tolist())
        raise

def main(models, X_train, y_train, latest_win_rate, latest_draw_rate):
    best_model = sorted(models, key=lambda x: x[1], reverse=True)[0][0]
    
    print("\nBest performing model:", type(best_model).__name__)
    print(f"Model accuracy: {sorted(models, key=lambda x: x[1], reverse=True)[0][1]:.2%}")
    
    if isinstance(best_model, LogisticRegression):
        print("\nAnalyzing Linear Regression Model...")
        coef_df, importance_df = analyze_lr_model(best_model, X_train.columns)
        
        print("\nLogistic Regression Formula for Win Probability:")
        print("-" * 50)
        win_formula = "P(Win) = 1 / (1 + e^-z), where z ="
        print(win_formula)
        
        for idx, feat in enumerate(X_train.columns):
            coef = best_model.coef_[2][idx]  
            if abs(coef) > 0.0001:  
                print(f"  {coef:+.2f} * {feat}")
        print(f"  {best_model.intercept_[2]:+.2f} (intercept)")
    
    while True:
        try:
            user_input = collect_user_input(latest_win_rate, latest_draw_rate, X_train)
            
            outcome = predict_match(best_model, user_input)
            print(f"\nPredicted Outcome: {outcome}")

            record = input("\nDo you want to record the actual result? (y/n): ").strip().lower()
            if record == 'y':
                actual_result = input("Enter the actual result (Win/Loss/Draw): ").strip().capitalize()
                result_map = {"Win": 2, "Loss": 0, "Draw": 1}
                actual_result_code = result_map.get(actual_result)
                
                if actual_result_code is not None:
                    X_train = pd.concat([X_train, user_input], ignore_index=True)
                    y_train = pd.concat([y_train, pd.Series([actual_result_code])], ignore_index=True)
                    
                    X_train = X_train.sort_values(by=['Date']).reset_index(drop=True)
                    latest_win_rate = X_train["WinRate"].iloc[-1]
                    latest_draw_rate = X_train["DrawRate"].iloc[-1]
                    
                    print("\nRetraining the model with new data...")
                    best_model.fit(X_train, y_train)
                    print("Model retrained successfully!")

                    if isinstance(best_model, LogisticRegression):
                        print("\nUpdated model analysis:")
                        coef_df, importance_df = analyze_lr_model(best_model, X_train.columns)
            
            cont = input("\nDo you want to predict another match? (y/n): ").strip().lower()
            if cont != 'y':
                break
                
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            cont = input("Do you want to try again? (y/n): ").strip().lower()
            if cont != 'y':
                break
    
    print("\nThank you for using the chess match predictor!")
