import config.db_config as db_config
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

password = quote_plus(db_config.DB_PASSWORD)
engine = create_engine(f'mysql+pymysql://{db_config.DB_USER}:{password}@{db_config.DB_HOST}:{db_config.DB_PORT}/{db_config.DB_NAME}')

print("🧹 Cleaning up database...")
with engine.begin() as conn:
    # Drop all DP-related tables
    tables = ['dp_measurement', 'dp_batch_member', 'dp_batch', 'dp_query_log', 'analyst_budget', 'census_income']
    
    for table in tables:
        try:
            conn.execute(text(f'DROP TABLE IF EXISTS {table}'))
            print(f"  ✓ Dropped {table}")
        except Exception as e:
            print(f"  ⚠️  Error dropping {table}: {e}")
    
    print("✅ Database cleaned!")
