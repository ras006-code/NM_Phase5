import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

class AISupplyChainManager:
    def __init__(self):
        # Simulated historical sales data
        self.historical_data = self._generate_historical_sales_data()

        # Train initial predictive model
        self.demand_forecast_model = self._train_demand_forecast_model()

    def _generate_historical_sales_data(self):
        """Generate synthetic historical sales data."""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', end='2024-04-30', freq='D')

        # Simulate sales with seasonal variations and trend
        base_sales = 100
        seasonal_variation = np.sin(np.arange(len(dates)) * (2 * np.pi / 365)) * 20
        trend = np.linspace(0, 30, len(dates))
        noise = np.random.normal(0, 10, len(dates))

        sales = base_sales + seasonal_variation + trend + noise
        sales = np.maximum(sales, 0)  # Ensure no negative sales

        df = pd.DataFrame({
            'date': dates,
            'sales': sales,
            'product_category': np.random.choice(['Electronics', 'Clothing', 'Home Goods'], len(dates))
        })
        return df

    def _train_demand_forecast_model(self):
        """Train a Random Forest Regressor for demand forecasting."""
        # Prepare features
        X = self.historical_data.copy()
        X['month'] = X['date'].dt.month
        X['day_of_week'] = X['date'].dt.dayofweek
        X = pd.get_dummies(X, columns=['product_category'])

        # Prepare features
        feature_columns = ['month', 'day_of_week'] + [col for col in X.columns if col.startswith('product_category_')]
        X_train = X[feature_columns]

        # Prepare target
        y_train = X['sales']

        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model

    def forecast_demand(self, days_ahead=30):
            """Generate demand forecast for the next 30 days."""
            forecast_dates = pd.date_range(start=datetime.now(), periods=days_ahead)

            forecast_data = []
            for category in ['Electronics', 'Clothing', 'Home Goods']:
                category_forecasts = []
                for date in forecast_dates:
                    # Prepare complete feature set, ensuring the order matches training data
                    feature_df = pd.DataFrame([{
                        'month': date.month,
                        'day_of_week': date.dayofweek,
                        'product_category_Electronics': 1 if category == 'Electronics' else 0,
                        'product_category_Clothing': 1 if category == 'Clothing' else 0,
                        'product_category_Home Goods': 1 if category == 'Home Goods' else 0
                    }])[self.demand_forecast_model.feature_names_in_] # Select columns in the correct order

                    # Predict
                    prediction = self.demand_forecast_model.predict(feature_df)[0]
                    category_forecasts.append({
                        'date': date,
                        'product_category': category,
                        'forecasted_sales': max(0, prediction)
                    })

                forecast_data.extend(category_forecasts)

            return pd.DataFrame(forecast_data)

    def inventory_optimization(self, forecast):
        """Simple inventory optimization based on demand forecast."""
        optimization_report = []

        for category in forecast['product_category'].unique():
            category_forecast = forecast[forecast['product_category'] == category]

            avg_daily_demand = category_forecast['forecasted_sales'].mean()
            total_forecast_demand = category_forecast['forecasted_sales'].sum()

            # Simple safety stock calculation
            safety_stock = avg_daily_demand * 1.5

            optimization_report.append({
                'product_category': category,
                'avg_daily_demand': round(avg_daily_demand, 2),
                'total_forecast_demand': round(total_forecast_demand, 2),
                'recommended_safety_stock': round(safety_stock, 2)
            })

        return pd.DataFrame(optimization_report)

def main():
    # Initialize Supply Chain Manager
    supply_chain_manager = AISupplyChainManager()

    # Generate Demand Forecast
    print("🔮 Demand Forecast for Next 30 Days:")
    demand_forecast = supply_chain_manager.forecast_demand()
    print(demand_forecast)
    print("\n")

    # Inventory Optimization
    print("📦 Inventory Optimization Recommendations:")
    inventory_recommendations = supply_chain_manager.inventory_optimization(demand_forecast)
    print(inventory_recommendations)

if __name__ == "__main__":
    main()
