#!/usr/bin/env python3
"""
Performance Test - Direct Implementation
Tests the differential privacy system performance using direct imports
"""

import time
import logging
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from config.db_config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_db_connection():
    """Create database connection."""
    import urllib.parse
    # URL encode the password to handle special characters
    password_encoded = urllib.parse.quote_plus(DB_PASSWORD)
    
    # Debug: Show the connection components
    print(f"Connecting to database:")
    print(f"  Host: {DB_HOST}")
    print(f"  Port: {DB_PORT}")
    print(f"  User: {DB_USER}")
    print(f"  Database: {DB_NAME}")
    print(f"  Password encoded: {password_encoded}")
    
    db_url = f"mysql+pymysql://{DB_USER}:{password_encoded}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    print(f"  Connection URL: mysql+pymysql://{DB_USER}:***@{DB_HOST}:{DB_PORT}/{DB_NAME}")
    
    engine = create_engine(db_url, pool_pre_ping=True)
    return engine

def load_test_data(engine, num_rows=100):
    """Load test data into database."""
    print(f"Loading {num_rows} rows of test data...")
    
    # Load data from CSV
    df = pd.read_csv('dataset/adult.csv').head(num_rows)
    df.columns = [col.strip().replace('-', '_').replace(' ', '_').replace('.', '_') for col in df.columns]
    
    print(f"Original data shape: {df.shape}")
    print(f"All columns: {list(df.columns)}")
    
    # Load to database
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS census_income"))
        df.to_sql('census_income', con=conn, if_exists='replace', index=False)
        
        # Verify the data was loaded
        result = conn.execute(text("SELECT COUNT(*) FROM census_income"))
        actual_count = result.fetchone()[0]
        
        # Show sample data
        sample_result = conn.execute(text("SELECT * FROM census_income LIMIT 3"))
        sample_rows = sample_result.fetchall()
        print(f"Sample data: {len(sample_rows)} rows")
        if sample_rows:
            print(f"First row preview: {sample_rows[0][:5]}...")  # Show first 5 columns
            
        # Show table structure
        desc_result = conn.execute(text("DESCRIBE census_income"))
        columns = desc_result.fetchall()
        print(f"Table columns: {[col[0] for col in columns]}")
    
    print(f"Data loaded successfully: {actual_count} rows")
    return actual_count

def add_laplace_noise(value, sensitivity, epsilon):
    """Add Laplace noise for differential privacy."""
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale)
    return value + noise

def execute_dp_query(engine, sql, epsilon=1.0, sensitivity=1.0):
    """Execute a differentially private query."""
    start_time = time.time()
    
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        true_value = result.fetchone()[0]
    
    # Handle None/NULL values
    if true_value is None:
        true_value = 0.0
    else:
        true_value = float(true_value)
    
    # Add differential privacy noise
    noisy_value = add_laplace_noise(true_value, sensitivity, epsilon)
    
    execution_time = time.time() - start_time
    
    return {
        'sql': sql,
        'true_result': true_value,
        'noisy_result': float(noisy_value),
        'noise_added': float(noisy_value - true_value),
        'epsilon': epsilon,
        'sensitivity': sensitivity,
        'execution_time_ms': execution_time * 1000
    }

def run_performance_test():
    """Run comprehensive performance test."""
    print("=== Differential Privacy System Performance Test ===\n")
    
    try:
        # Create database connection
        engine = create_db_connection()
        print("Database connection established successfully")
        
        # Load test data
        row_count = load_test_data(engine, num_rows=1000)
        
        # Test queries with proper column values
        test_queries = [
            ("SELECT COUNT(*) FROM census_income", "Total count", 1.0),
            ("SELECT COUNT(*) FROM census_income WHERE age > 30", "Count age > 30", 1.0),
            ("SELECT AVG(age) FROM census_income", "Average age", 5.0),
            ("SELECT COUNT(*) FROM census_income WHERE sex = 'Male'", "Count males", 1.0),
            ("SELECT COUNT(*) FROM census_income WHERE income = ' >50K'", "Count high income", 1.0),
            ("SELECT AVG(hours_per_week) FROM census_income", "Average hours", 10.0),
        ]
        
        # First, let's check what data we actually have
        print(f"\nData verification:")
        with engine.connect() as conn:
            # Check total count
            result = conn.execute(text("SELECT COUNT(*) FROM census_income"))
            total_count = result.fetchone()[0]
            print(f"  Total rows: {total_count}")
            
            if total_count > 0:
                # Check age distribution
                result = conn.execute(text("SELECT MIN(age), MAX(age), AVG(age) FROM census_income"))
                min_age, max_age, avg_age = result.fetchone()
                print(f"  Age range: {min_age or 'NULL'} to {max_age or 'NULL'}, avg: {avg_age:.1f if avg_age else 'NULL'}")
                
                # Check sex values
                result = conn.execute(text("SELECT DISTINCT sex FROM census_income LIMIT 5"))
                sex_values = [row[0] for row in result.fetchall()]
                print(f"  Sex values: {sex_values}")
                
                # Check income values  
                result = conn.execute(text("SELECT DISTINCT income FROM census_income LIMIT 5"))
                income_values = [row[0] for row in result.fetchall()]
                print(f"  Income values: {income_values}")
                
                # Check hours_per_week
                result = conn.execute(text("SELECT MIN(hours_per_week), MAX(hours_per_week), AVG(hours_per_week) FROM census_income"))
                min_hours, max_hours, avg_hours = result.fetchone()
                print(f"  Hours range: {min_hours or 'NULL'} to {max_hours or 'NULL'}, avg: {avg_hours:.1f if avg_hours else 'NULL'}")
            else:
                print("  ⚠️  WARNING: No data found in table! Data loading may have failed.")
                # Try to reload data with debugging
                print("  Attempting to reload data...")
                conn.execute(text("DROP TABLE IF EXISTS census_income"))
                conn.commit()
                
                # Reload with explicit commit
                df = pd.read_csv('dataset/adult.csv').head(100)  # Use smaller dataset for debugging
                df.columns = [col.strip().replace('-', '_').replace(' ', '_').replace('.', '_') for col in df.columns]
                df.to_sql('census_income', con=conn, if_exists='replace', index=False)
                conn.commit()
                
                # Recheck count
                result = conn.execute(text("SELECT COUNT(*) FROM census_income"))
                total_count = result.fetchone()[0]
                print(f"  After reload - Total rows: {total_count}")
        
        print(f"\nExecuting {len(test_queries)} differential privacy queries:")
        print("-" * 80)
        
        results = []
        total_privacy_budget = 0
        
        for i, (sql, description, sensitivity) in enumerate(test_queries, 1):
            epsilon = 0.5  # Privacy budget per query
            
            print(f"\nQuery {i}: {description}")
            print(f"SQL: {sql}")
            print(f"Privacy parameters: ε={epsilon}, Δ={sensitivity}")
            
            result = execute_dp_query(engine, sql, epsilon, sensitivity)
            results.append(result)
            total_privacy_budget += epsilon
            
            print(f"True result: {result['true_result']:.2f}")
            print(f"Noisy result: {result['noisy_result']:.2f}")
            print(f"Noise added: {result['noise_added']:.2f}")
            print(f"Execution time: {result['execution_time_ms']:.2f} ms")
        
        # Performance summary
        print(f"\n{'='*60}")
        print("PERFORMANCE SUMMARY")
        print(f"{'='*60}")
        print(f"Total queries executed: {len(results)}")
        print(f"Total privacy budget used: {total_privacy_budget:.2f}")
        print(f"Average execution time: {np.mean([r['execution_time_ms'] for r in results]):.2f} ms")
        print(f"Total execution time: {sum(r['execution_time_ms'] for r in results):.2f} ms")
        print(f"Data rows processed: {row_count}")
        
        # Accuracy analysis
        print(f"\nACCURACY ANALYSIS")
        print("-" * 30)
        for i, result in enumerate(results, 1):
            relative_error = abs(result['noise_added']) / abs(result['true_result']) * 100 if result['true_result'] != 0 else 0
            print(f"Query {i}: {relative_error:.1f}% relative error")
        
        avg_relative_error = np.mean([
            abs(r['noise_added']) / abs(r['true_result']) * 100 
            for r in results if r['true_result'] != 0
        ])
        print(f"Average relative error: {avg_relative_error:.1f}%")
        
        print(f"\n{'='*60}")
        print("✅ Performance test completed successfully!")
        print(f"{'='*60}")
        
        return results
        
    except Exception as e:
        logger.error(f"Performance test failed: {e}")
        print(f"❌ Error: {e}")
        return None
    
    finally:
        if 'engine' in locals():
            engine.dispose()

if __name__ == "__main__":
    results = run_performance_test()
    if results:
        print(f"\nTest completed with {len(results)} successful queries")
    else:
        print("Test failed")
