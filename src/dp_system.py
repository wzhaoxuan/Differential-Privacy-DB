import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from sqlalchemy import create_engine, text
import numpy as np
import time
from urllib.parse import quote_plus
from typing import Optional, List, Tuple, Dict, Any

import config.db_config as db_config

# Shared query configuration - used by both main system and performance assessment
DEFAULT_TEST_QUERIES = [
    # 2 COUNT queries
    "SELECT COUNT(*) FROM census_income",
    "SELECT COUNT(*) FROM census_income WHERE `marital.status`='Divorced'",
    
    # 2 SUM queries  
    "SELECT SUM(`hours.per.week`) FROM census_income",
    "SELECT SUM(age) FROM census_income WHERE `workclass`='Private'",
    
    # 2 AVG queries
    "SELECT AVG(age) FROM census_income", 
    "SELECT AVG(`hours.per.week`) FROM census_income WHERE `education`='Bachelors'",
    
    # 2 BATCH queries (UNION operations)
    """SELECT COUNT(*) as count_high_income FROM census_income WHERE `income`='>50K'
        UNION
        SELECT COUNT(*) as count_low_income FROM census_income WHERE `income`='<=50K'""",
            
    """SELECT AVG(age) as avg_age_male FROM census_income WHERE `sex`='Male'
        UNION  
        SELECT AVG(age) as avg_age_female FROM census_income WHERE `sex`='Female'"""
]


class DatabaseManager:
    """Handles database connection and basic operations."""
    
    def __init__(self):
        self.engine = self._create_engine()
    
    def _create_engine(self):
        """Create SQLAlchemy engine with database configuration."""
        password = quote_plus(db_config.DB_PASSWORD)
        return create_engine(
            f"mysql+pymysql://{db_config.DB_USER}:"
            f"{password}@"
            f"{db_config.DB_HOST}:"
            f"{db_config.DB_PORT}/"
            f"{db_config.DB_NAME}"
        )
    
    def get_connection(self):
        """Get database connection."""
        return self.engine.begin()
    
    def execute_query(self, query: str, params: Optional[Tuple] = None):
        """Execute a query and return results."""
        with self.get_connection() as conn:
            if params:
                return conn.execute(text(query), params)
            return conn.execute(text(query))


class DataLoader:
    """Handles loading and preparing data."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def load_and_prepare_data(self, csv_path: str = 'dataset/adult.csv', 
                            num_rows: int = 100) -> pd.DataFrame:
        """Load and sanitize data from CSV."""
        df = (
            pd.read_csv(csv_path)
              .head(num_rows)
              .rename(columns=lambda c: c.strip().replace('-', '_').replace(' ', '_'))
        )
        return df
    
    def load_data_to_database(self, df: pd.DataFrame, table_name: str = 'census_income'):
        """Load DataFrame to database table."""
        with self.db_manager.get_connection() as conn:
            # Drop table if it exists
            conn.execute(text(f"DROP TABLE IF EXISTS {table_name};"))
            
            # Create and populate table
            df.to_sql(
                name=table_name,
                con=conn,
                if_exists='replace',
                index=False,
                dtype={}
            )
        
        print(f"{table_name} table created and {len(df)} rows inserted into {db_config.DB_NAME}")


class SchemaManager:
    """Manages database schema creation for differential privacy system."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def deploy_dp_schema(self):
        """Deploy differential privacy schema tables."""
        ddl_statements = self._get_ddl_statements()
        
        with self.db_manager.get_connection() as conn:
            for statement in ddl_statements:
                conn.execute(text(statement))
            
            # Initialize default analyst budget - use INSERT IGNORE for MySQL
            conn.execute(text("""
                INSERT IGNORE INTO analyst_budget (analyst_id, epsilon_total) 
                VALUES ('analyst_1', 1.0)
            """))
            conn.commit()
    
    def _get_ddl_statements(self) -> List[str]:
        """Get DDL statements for creating DP schema."""
        return [
            """
            CREATE TABLE IF NOT EXISTS analyst_budget (
              analyst_id   VARCHAR(255) PRIMARY KEY,
              epsilon_total DECIMAL(10,6) NOT NULL,
              epsilon_spent DECIMAL(10,6) NOT NULL DEFAULT 0
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS dp_query_log (
              query_id      INT AUTO_INCREMENT PRIMARY KEY,
              analyst_id    VARCHAR(255),
              raw_sql       TEXT NOT NULL,
              canonical_sql TEXT,
              query_type    ENUM('single','mean','batch'),
              delta_sensitivity DECIMAL(10,6) NOT NULL,
              epsilon_charge DECIMAL(10,6) NOT NULL,
              mechanism     VARCHAR(255),
              noise_seed    VARCHAR(255),
              noisy_result  DECIMAL(15,6),
              status        ENUM('PENDING','DONE','FAILED') DEFAULT 'PENDING',
              created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS dp_batch (
              batch_id      INT AUTO_INCREMENT PRIMARY KEY,
              analyst_id    VARCHAR(255),
              epsilon_charge DECIMAL(10,6) NOT NULL,
              status        ENUM('pending','executed','cancelled') DEFAULT 'pending',
              created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
              executed_at   TIMESTAMP NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS dp_batch_member (
              batch_id INT, 
              query_id INT, 
              PRIMARY KEY(batch_id, query_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS dp_measurement (
              measurement_id INT AUTO_INCREMENT PRIMARY KEY,
              batch_id       INT,
              vector_index   INT,
              noisy_value    DECIMAL(15,6),
              created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        ]


class QueryParser:
    """Handles SQL parsing and sensitivity analysis."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.epsilon_map = {'single': 0.05, 'mean': 0.1, 'batch': 0.2}
    
    def compute_sensitivity(self, sql: str) -> float:
        """Compute sensitivity for a given SQL query."""
        sql_lower = sql.lower()
        if sql_lower.startswith("select count"): 
            return 1.0
        if sql_lower.startswith("select sum"):   
            return 1.0
        if sql_lower.startswith("select avg"):   
            return 1.0  # assume range/100
        return 1.0
    
    def classify_query(self, sql: str) -> str:
        """Classify query type."""
        sql_lower = sql.lower()

        # Check for batch operations FIRST (highest priority)
        if 'union' in sql_lower or ';' in sql_lower:   
            return 'batch'
        
        # Check for mean queries
        if sql_lower.startswith("select avg"): 
            return 'mean'
        
        # Default to single query
        return 'single'
    
    def parse_and_log(self, raw_sql: str, analyst_id: str = 'analyst_1') -> int:
        """Parse SQL query and log it to the database."""
        canonical_sql = raw_sql.strip().upper()
        delta_sensitivity = self.compute_sensitivity(canonical_sql)
        query_type = self.classify_query(canonical_sql)
        epsilon_charge = self.epsilon_map[query_type]
        
        with self.db_manager.get_connection() as conn:
            result = conn.execute(text("""
              INSERT INTO dp_query_log
                (analyst_id, raw_sql, canonical_sql, query_type, delta_sensitivity, epsilon_charge)
              VALUES (:analyst_id, :raw_sql, :canonical_sql, :query_type, :delta_sensitivity, :epsilon_charge)
            """), {
                'analyst_id': analyst_id,
                'raw_sql': raw_sql,
                'canonical_sql': canonical_sql,
                'query_type': query_type,
                'delta_sensitivity': delta_sensitivity,
                'epsilon_charge': epsilon_charge
            })
            
            query_id = result.lastrowid
            conn.commit()
            return query_id


class PrivacyEngine:
    """Implements differential privacy mechanisms."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def laplace_noise(self, scale: float) -> float:
        """Generate Laplace noise."""
        return np.random.laplace(0, scale)
    
    def process_pending_queries(self, analyst_id: str = 'analyst_1'):
        """Process all pending queries for an analyst."""
        with self.db_manager.get_connection() as conn:
            result = conn.execute(text("""
                SELECT query_id, raw_sql, query_type, delta_sensitivity, epsilon_charge 
                FROM dp_query_log 
                WHERE status='PENDING' AND analyst_id=:analyst_id
            """), {'analyst_id': analyst_id})
            
            for row in result:
                query_id, raw_sql, query_type, delta_sensitivity, epsilon_charge = row
                if not self._check_and_update_budget(conn, analyst_id, epsilon_charge):
                    self._mark_query_failed(conn, query_id)
                    continue
                
                if query_type == 'single':
                    self._process_single_query(conn, query_id, raw_sql, delta_sensitivity, epsilon_charge)
                elif query_type == 'mean':
                    self._process_mean_query(conn, query_id, raw_sql, delta_sensitivity, epsilon_charge)
                else:  # batch
                    self._defer_to_batch(conn, analyst_id, query_id, epsilon_charge)
            
            conn.commit()
    
    def _check_and_update_budget(self, conn, analyst_id: str, epsilon_charge: float) -> bool:
        """Check if analyst has enough budget and update if so."""

        # Convert to float to ensure consistent type handling
        epsilon_charge_float = float(epsilon_charge)

        result = conn.execute(text("""
          UPDATE analyst_budget
             SET epsilon_spent = epsilon_spent + :epsilon_charge
           WHERE analyst_id = :analyst_id
             AND epsilon_spent + :epsilon_charge <= epsilon_total
        """), {
            'epsilon_charge': epsilon_charge_float,
            'analyst_id': analyst_id
        })
        
        return result.rowcount > 0
    
    def _mark_query_failed(self, conn, query_id: int):
        """Mark a query as failed."""
        conn.execute(text("UPDATE dp_query_log SET status='FAILED' WHERE query_id=:query_id"), 
                    {'query_id': query_id})
    
    def _process_single_query(self, conn, query_id: int, raw_sql: str, 
                            delta_sensitivity: float, epsilon_charge: float):
        """Process a single query with Laplace mechanism."""
        # Extract the actual query part (remove SELECT and FROM parts for execution)
        query_parts = raw_sql.upper().split()
        if len(query_parts) >= 2:
            if 'COUNT(*)' in raw_sql.upper():
                actual_query = raw_sql
            else:
                actual_query = raw_sql
        else:
            actual_query = raw_sql
            
        result = conn.execute(text(actual_query))
        row = result.fetchone()
        value = float(row[0]) if row and row[0] is not None else 0
        noisy_result = value + self.laplace_noise(delta_sensitivity / epsilon_charge)
        
        conn.execute(text("""
          UPDATE dp_query_log
             SET noisy_result=:noisy_result, mechanism='LAPLACE', status='DONE'
           WHERE query_id=:query_id
        """), {'noisy_result': noisy_result, 'query_id': query_id})
    
    def _process_mean_query(self, conn, query_id: int, raw_sql: str, 
                          delta_sensitivity: float, epsilon_charge: float):
        """Process a mean query with instantaneous mechanism."""
        result = conn.execute(text(raw_sql))
        row = result.fetchone()
        value = float(row[0]) if row and row[0] is not None else 0
        noisy_result = value + self.laplace_noise(delta_sensitivity / epsilon_charge)
        
        conn.execute(text("""
          UPDATE dp_query_log
             SET noisy_result=:noisy_result, mechanism='INST_MEAN', status='DONE'
           WHERE query_id=:query_id
        """), {'noisy_result': noisy_result, 'query_id': query_id})
    
    def _defer_to_batch(self, conn, analyst_id: str, query_id: int, epsilon_charge: float):
        """Defer query to batch processing."""
        result = conn.execute(text("INSERT INTO dp_batch (analyst_id, epsilon_charge) VALUES (:analyst_id, :epsilon_charge)"), 
                             {'analyst_id': analyst_id, 'epsilon_charge': epsilon_charge})
        batch_id = result.lastrowid
        
        conn.execute(text("INSERT INTO dp_batch_member(batch_id, query_id) VALUES (:batch_id, :query_id)"), 
                    {'batch_id': batch_id, 'query_id': query_id})
        conn.execute(text("UPDATE dp_query_log SET status='DONE', mechanism='BATCH_PENDING' WHERE query_id=:query_id"), 
                    {'query_id': query_id})


class HolisticProcessor:
    """Handles batch processing with holistic privacy mechanisms."""
    
    def __init__(self, db_manager: DatabaseManager, privacy_engine: PrivacyEngine):
        self.db_manager = db_manager
        self.privacy_engine = privacy_engine
    
    def run_holistic_processing(self):
        """Process all pending batches with holistic mechanisms."""
        with self.db_manager.get_connection() as conn:
            result = conn.execute(text("SELECT batch_id, epsilon_charge FROM dp_batch WHERE status='pending'"))
            
            for row in result:
                batch_id, epsilon_charge = row
                self._process_batch(conn, batch_id, epsilon_charge)
            
            conn.commit()
    
    def _process_batch(self, conn, batch_id: int, epsilon_charge: float):
        """Process a single batch."""
        # Get batch members
        result = conn.execute(text("""
          SELECT q.query_id, q.raw_sql
            FROM dp_batch_member m
            JOIN dp_query_log q ON m.query_id=q.query_id
           WHERE m.batch_id=:batch_id
        """), {'batch_id': batch_id})
        
        members = list(result)

        # Convert epsilon_charge to float to avoid decimal/float operation issues
        epsilon_charge_float = float(epsilon_charge)
        
        # Process each query in the batch with identity strategy
        for idx, (query_id, raw_sql) in enumerate(members):
            query_result = conn.execute(text(raw_sql))
            row = query_result.fetchone()
            value = float(row[0]) if row and row[0] is not None else 0
            # Use converted float value for division
            noisy_result = value + self.privacy_engine.laplace_noise(1.0 / epsilon_charge_float)
            
            # Store measurement
            conn.execute(text("""
                INSERT INTO dp_measurement(batch_id, vector_index, noisy_value) 
                VALUES (:batch_id, :vector_index, :noisy_value)
            """), {'batch_id': batch_id, 'vector_index': idx, 'noisy_value': noisy_result})
            
            # Update query log
            conn.execute(text("""
              UPDATE dp_query_log
                 SET noisy_result=:noisy_result, mechanism='HOLISTIC', status='DONE'
               WHERE query_id=:query_id
            """), {'noisy_result': noisy_result, 'query_id': query_id})
        
        # Mark batch as executed
        conn.execute(text("""
            UPDATE dp_batch 
            SET status='executed', executed_at=CURRENT_TIMESTAMP 
            WHERE batch_id=:batch_id
        """), {'batch_id': batch_id})


class DifferentialPrivacySystem:
    """Main orchestrator class for the differential privacy system."""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.data_loader = DataLoader(self.db_manager)
        self.schema_manager = SchemaManager(self.db_manager)
        self.query_parser = QueryParser(self.db_manager)
        self.privacy_engine = PrivacyEngine(self.db_manager)
        self.holistic_processor = HolisticProcessor(self.db_manager, self.privacy_engine)
    
    def initialize_system(self, csv_path: str = 'dataset/adult.csv', num_rows: int = 100):
        """Initialize the entire differential privacy system."""
        # Load data
        df = self.data_loader.load_and_prepare_data(csv_path, num_rows)
        self.data_loader.load_data_to_database(df)
        
        # Deploy schema
        self.schema_manager.deploy_dp_schema()
    
    def submit_query(self, sql: str, analyst_id: str = 'analyst_1') -> int:
        """Submit a query for differential privacy processing."""
        return self.query_parser.parse_and_log(sql, analyst_id)

    def process_queries(self, analyst_id: str = 'analyst_1'):
        """Process all pending queries."""
        self.privacy_engine.process_pending_queries(analyst_id)
    
    def process_batches(self):
        """Process all pending batches."""
        self.holistic_processor.run_holistic_processing()
    
    def get_query_results(self) -> pd.DataFrame:
        """Get all query results."""
        with self.db_manager.get_connection() as conn:
            return pd.read_sql("SELECT * FROM dp_query_log", conn)
    
    def get_specific_query_results(self, query_ids: List[int]) -> pd.DataFrame:
        """Get results for specific query IDs only."""
        if not query_ids:
            return pd.DataFrame()
        
        query_ids_str = ','.join(map(str, query_ids))
        with self.db_manager.get_connection() as conn:
            return pd.read_sql(
                f"SELECT * FROM dp_query_log WHERE query_id IN ({query_ids_str}) ORDER BY query_id", 
                conn
            )
    
    def clear_previous_results(self):
        """Clear previous query logs for a fresh start."""
        try:
            with self.db_manager.get_connection() as conn:
                conn.execute(text("DELETE FROM dp_query_log"))
                conn.execute(text("DELETE FROM dp_batch"))
                conn.execute(text("DELETE FROM dp_batch_member"))
                conn.execute(text("DELETE FROM dp_measurement"))
                conn.execute(text("UPDATE analyst_budget SET epsilon_spent = 0.0 WHERE analyst_id = 'analyst_1'"))
                conn.commit()
                print("🧹 Cleared previous query logs and reset privacy budget")
        except Exception as e:
            print(f"Warning: Could not clear previous logs: {e}")
    
    def get_default_test_queries(self) -> List[str]:
        """Get the default test queries used by the system."""
        return DEFAULT_TEST_QUERIES.copy()
    
    def run_default_test(self):
        """Run the default test queries."""
        print("Submitting queries...")
        query_ids = []
        
        # Submit queries and track their IDs
        for i, query in enumerate(DEFAULT_TEST_QUERIES, 1):
            query_id = self.submit_query(query)
            query_ids.append(query_id)
        
        print(f"Submitted queries: {', '.join(map(str, query_ids))}")
        
        # Process queries
        print("Processing with differential privacy...")
        self.process_queries()
        time.sleep(1)
        self.process_batches()
        
        # View results - Get only the current run's results
        print("Getting results...")
        results = self.get_specific_query_results(query_ids)
        print("DIFFERENTIAL PRIVACY RESULTS:")
        print("="*50)
        if not results.empty:
            print(results[['query_id', 'query_type', 'mechanism', 'noisy_result', 'epsilon_charge', 'status']].to_string(index=False))
        else:
            print("No results found")
        print("="*50)
        
        return results


# Usage example
if __name__ == "__main__":
    # Initialize system
    dp_system = DifferentialPrivacySystem()
    dp_system.initialize_system()
    
    # Optional: Clear previous results for a fresh start
    dp_system.clear_previous_results()
    
    # Run default test
    dp_system.run_default_test()
