#!/usr/bin/env python3
"""Generate synthetic business data for the BI system."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed import set_seed
from src.utils.io import PathManager


def generate_customers(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic customer data.
    
    Args:
        n: Number of customers to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with customer data
    """
    set_seed(seed)
    
    states = ['CA', 'TX', 'NY', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI']
    segments = ['Consumer', 'Corporate', 'Home Office']
    
    customers = {
        'customer_id': range(1, n + 1),
        'customer_name': [f"Customer_{i:04d}" for i in range(1, n + 1)],
        'segment': np.random.choice(segments, n, p=[0.6, 0.25, 0.15]),
        'state': np.random.choice(states, n, p=[0.15, 0.12, 0.11, 0.10, 0.09, 
                                                 0.08, 0.08, 0.08, 0.09, 0.10]),
        'signup_date': [
            datetime(2020, 1, 1) + timedelta(days=int(x))
            for x in np.random.randint(0, 1095, n)
        ]
    }
    
    return pd.DataFrame(customers)


def generate_products(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic product data.
    
    Args:
        n: Number of products to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with product data
    """
    set_seed(seed + 1)
    
    categories = ['Technology', 'Furniture', 'Office Supplies']
    
    products = {
        'product_id': range(1, n + 1),
        'product_name': [f"Product_{cat[0]}{i:03d}" for i, cat in 
                        zip(range(1, n + 1), np.random.choice(categories, n))],
        'category': np.random.choice(categories, n, p=[0.4, 0.3, 0.3]),
        'cost': np.random.uniform(5, 500, n).round(2)
    }
    
    return pd.DataFrame(products)


def generate_orders(n: int = 10000, n_customers: int = 1000, 
                   n_products: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic order data.
    
    Args:
        n: Number of orders to generate
        n_customers: Number of unique customers
        n_products: Number of unique products
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with order data
    """
    set_seed(seed + 2)
    
    regions = ['East', 'West', 'Central', 'South']
    
    # Generate order dates with seasonal pattern
    base_date = datetime(2022, 1, 1)
    dates = []
    for i in range(n):
        days_offset = int(np.random.exponential(365))
        if days_offset > 1095:  # Cap at 3 years
            days_offset = np.random.randint(0, 1095)
        dates.append(base_date + timedelta(days=days_offset))
    
    # Price markup: 1.5x - 3x of cost
    markups = np.random.uniform(1.5, 3.0, n)
    
    # Sample product costs (simplified - in reality we'd join, but for generation...)
    product_costs = np.random.uniform(5, 500, n)
    
    orders = {
        'order_id': range(1, n + 1),
        'order_date': dates,
        'customer_id': np.random.randint(1, n_customers + 1, n),
        'product_id': np.random.randint(1, n_products + 1, n),
        'quantity': np.random.choice([1, 2, 3, 4, 5, 10], n, p=[0.4, 0.25, 0.15, 0.1, 0.05, 0.05]),
        'price': (product_costs * markups).round(2),
        'region': np.random.choice(regions, n, p=[0.3, 0.25, 0.25, 0.2])
    }
    
    df = pd.DataFrame(orders)
    df = df.sort_values('order_date').reset_index(drop=True)
    
    return df


def main():
    """Generate all synthetic datasets."""
    print(">> Generating synthetic BI data...")
    
    path_mgr = PathManager()
    
    # Generate datasets
    print("  >> Generating customers...")
    customers = generate_customers(n=1000, seed=42)
    customers.to_csv(path_mgr.data_raw / "customers.csv", index=False)
    print(f"     [OK] Created {len(customers)} customers")
    
    print("  >> Generating products...")
    products = generate_products(n=200, seed=42)
    products.to_csv(path_mgr.data_raw / "products.csv", index=False)
    print(f"     [OK] Created {len(products)} products")
    
    print("  >> Generating orders...")
    orders = generate_orders(n=10000, n_customers=1000, n_products=200, seed=42)
    orders.to_csv(path_mgr.data_raw / "orders.csv", index=False)
    print(f"     [OK] Created {len(orders)} orders")
    
    # Summary statistics
    print("\n>> Data Summary:")
    print(f"  Date range: {orders['order_date'].min()} to {orders['order_date'].max()}")
    print(f"  Total revenue: ${(orders['quantity'] * orders['price']).sum():,.2f}")
    print(f"  Avg order value: ${(orders['quantity'] * orders['price']).mean():,.2f}")
    print(f"  Regions: {orders['region'].nunique()}")
    print(f"  Customer segments: {customers['segment'].nunique()}")
    
    print("\n[SUCCESS] Data generation complete!")


if __name__ == "__main__":
    main()

