"""
Check column names in the census_income table
"""
from dp_system import DifferentialPrivacySystem

# Initialize system
dp_system = DifferentialPrivacySystem()
dp_system.initialize_system(num_rows=10)

# Check columns
with dp_system.db_manager.get_connection() as conn:
    from sqlalchemy import text
    result = conn.execute(text("DESCRIBE census_income"))
    columns = result.fetchall()
    
    print("Available columns in census_income table:")
    for col in columns:
        print(f"  - {col[0]} ({col[1]})")
