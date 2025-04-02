import matplotlib.pyplot as plt #type:ignore
import seaborn as sns #type:ignore

def plot_sales_distribution(df):
    plt.figure(figsize=(8,5))
    sns.histplot(df['Item_Outlet_Sales'], bins=30, kde=True)
    plt.title('Sales Distribution')
    plt.show()
