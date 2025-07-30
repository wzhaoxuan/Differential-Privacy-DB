#!/usr/bin/env python3
"""Debug script to check database content"""

import pandas as pd
from sqlalchemy import create_engine, text
import urllib.parse
from config.db_config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME

def main():
    # Create connection
    password_encoded = urllib.parse.quote_plus(DB_PASSWORD)
    db_url = f'mysql+pymysql://{DB_USER}:{password_encoded}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
    engine = create_engine(db_url)

    with engine.connect() as conn:
        print('=== Table Structure ===')
        result = conn.execute(text('SHOW COLUMNS FROM census_income'))
        for row in result:
            print(f'{row[0]:20s} {row[1]}')
        
        print('\n=== Sample Data ===')
        result = conn.execute(text('SELECT * FROM census_income LIMIT 3'))
        for i, row in enumerate(result):
            print(f'Row {i+1}: {list(row)[:8]}')
        
        print('\n=== Data Counts ===')
        queries = [
            ('Total rows', 'SELECT COUNT(*) FROM census_income'),
            ('Age over 30', 'SELECT COUNT(*) FROM census_income WHERE age > 30'), 
            ('Males', 'SELECT COUNT(*) FROM census_income WHERE sex = "Male"'),
            ('Average age', 'SELECT AVG(age) FROM census_income'),
            ('High income', 'SELECT COUNT(*) FROM census_income WHERE income = " >50K"'),
            ('Age range', 'SELECT MIN(age), MAX(age) FROM census_income'),
            ('Income values', 'SELECT DISTINCT income FROM census_income LIMIT 5')
        ]
        
        for name, query in queries:
            try:
                result = conn.execute(text(query))
                row = result.fetchone()
                print(f'{name}: {row[0] if len(row) == 1 else row}')
            except Exception as e:
                print(f'{name}: ERROR - {e}')

    engine.dispose()

if __name__ == "__main__":
    main()
