from src.data_preprocessing import load_data, clean_data
from src.eda import plot_sales_distribution
from src.model import train_model
from src.predict import save_predictions

def main():
    # Load & Clean Data
    train, test = load_data()
    train = clean_data(train)
    test = clean_data(test)

    # EDA
    plot_sales_distribution(train)

    # Feature & Target
    X = train.drop(['Item_Outlet_Sales'], axis=1).select_dtypes(include='number')
    y = train['Item_Outlet_Sales']
    X_test = test.select_dtypes(include='number')

    # Model Training
    model = train_model(X, y)

    # Prediction
    predictions = model.predict(X_test)
    save_predictions(test, predictions)

if __name__ == "__main__":
    main()
