import pandas as pd
import config.db_config as db_config
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

# Load and check data
print("📊 Loading dataset...")
df = pd.read_csv('dataset/adult.csv').head(10)
print("Original columns:", df.columns.tolist())

# Apply the same transformation as in DataLoader
df_clean = df.rename(columns=lambda c: c.strip().replace('-', '_').replace(' ', '_'))
print("Cleaned columns:", df_clean.columns.tolist())

# Connect to database
password = quote_plus(db_config.DB_PASSWORD)
engine = create_engine(f'mysql+pymysql://{db_config.DB_USER}:{password}@{db_config.DB_HOST}:{db_config.DB_PORT}/{db_config.DB_NAME}')

# Check what's actually in the database
print("\n🔍 Checking database table...")
with engine.begin() as conn:
    try:
        result = conn.execute(text('DESCRIBE census_income'))
        print("Table columns:")
        for row in result:
            print(f"  {row[0]} - {row[1]}")
    except Exception as e:
        print(f"Error: {e}")
        
    try:
        result = conn.execute(text('SELECT * FROM census_income LIMIT 1'))
        row = result.fetchone()
        if row:
            print(f"\nSample row keys: {row.keys()}")
    except Exception as e:
        print(f"Sample query error: {e}")
