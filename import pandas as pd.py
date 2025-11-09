import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

try:
    data = pd.read_csv('CakePricing.csv')
    cake_types = [
        ('chocolate', 'Demand for chocolate', ['Price of chocolate', 'Price for vanilla', 'Price for marble']),
        ('vanilla', 'Demand for vanilla', ['Price for vanilla', 'Price of chocolate', 'Price for marble']),
        ('marble', 'Demand for marble', ['Price for marble', 'Price of chocolate', 'Price for vanilla'])
    ]

    for cake, demand_col, price_cols in cake_types:
        X = data[price_cols].values
        y = data[demand_col].values
        model = LinearRegression()
        model.fit(X, y)
        intercept = round(model.intercept_, 2)
        coefs = [round(coef, 2) for coef in model.coef_]
        r2 = r2_score(y, model.predict(X))
        equation = f"Demand equation for {cake} cakes: Demand = {intercept} "
        for i, coef in enumerate(coefs):
            equation += f"+ ({coef}) * {price_cols[i]} "
        print(f"{equation}(R-squared = {r2:.2f})")

        # Statistical validation
        X_sm = sm.add_constant(data[price_cols])
        model_sm = sm.OLS(y, X_sm).fit()
        print(f"\nResults for {cake} cakes:")
        print(model_sm.summary())

        # Visuals
        plt.scatter(data[price_cols[0]], y, label=f'{cake} Demand', alpha=0.5)
        plt.plot(data[price_cols[0]], model.predict(X), color='red', label='Fit')
        plt.xlabel(price_cols[0])
        plt.ylabel(demand_col)
        plt.title(f'{cake} Demand vs. {price_cols[0]}')
        plt.legend()
        plt.show()

        # Residual plot
        residuals = y - model.predict(X)
        plt.scatter(model.predict(X), residuals, alpha=0.5)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('Predicted Demand')
        plt.ylabel('Residuals')
        plt.title(f'Residual Plot for {cake} Cakes')
        plt.show()

except FileNotFoundError:
    print("Error: File 'CakePricing.csv' not found.")