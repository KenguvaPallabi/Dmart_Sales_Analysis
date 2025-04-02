import pandas as pd

def save_predictions(test, predictions):
    test['Predicted_Sales'] = predictions
    test[['Item_Identifier', 'Outlet_Identifier', 'Predicted_Sales']].to_csv('data/predictions.csv', index=False)
    print("Predictions saved to data/predictions.csv")
