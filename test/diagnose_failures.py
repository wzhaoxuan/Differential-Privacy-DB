"""
Diagnose why DP queries are failing and how to prevent None/NaN/FAILED results
"""

import pandas as pd
from sqlalchemy import create_engine, text
import sys

# Import DP system
exec(open('dp_system.py', 'r', encoding='utf-8').read(), globals())

def diagnose_system():
    """Diagnose the current system state"""
    
    print("🔍 DIAGNOSING DIFFERENTIAL PRIVACY SYSTEM")
    print("=" * 50)
    
    try:
        dp_system = DifferentialPrivacySystem()
        
        with dp_system.db_manager.get_connection() as conn:
            
            # 1. Check privacy budget
            print("\n💰 PRIVACY BUDGET STATUS:")
            print("-" * 25)
            budget_result = conn.execute(text("""
                SELECT analyst_id, epsilon_total, epsilon_spent, 
                       (epsilon_total - epsilon_spent) as remaining
                FROM analyst_budget
            """))
            
            for row in budget_result:
                print(f"Analyst: {row[0]}")
                print(f"Total Budget: {row[1]}")
                print(f"Spent: {row[2]}")
                print(f"Remaining: {row[3]}")
                
            # 2. Check query log summary
            print("\n📊 QUERY LOG SUMMARY:")
            print("-" * 20)
            log_result = conn.execute(text("""
                SELECT status, mechanism, COUNT(*) as count
                FROM dp_query_log 
                GROUP BY status, mechanism
                ORDER BY status, mechanism
            """))
            
            for row in log_result:
                print(f"Status: {row[0]}, Mechanism: {row[1]}, Count: {row[2]}")
                
    except Exception as e:
        print(f"Error during diagnosis: {e}")
        import traceback
        traceback.print_exc()

def get_prevention_strategies():
    """Provide strategies to prevent failures"""
    
    print("\n\n🛠️  PREVENTION STRATEGIES")
    print("=" * 50)
    
    print("""
1. 📊 INCREASE PRIVACY BUDGET:
   - Current budget appears to be 1.0 epsilon
   - Consider increasing to 2.0 or 3.0 for more queries
   - Update in dp_system.py: analyst_budget table
   
2. 🔄 RESET BUDGET BETWEEN TESTS:
   - Clear dp_query_log table before each test
   - Reset epsilon_spent to 0
   - This prevents budget exhaustion from previous runs
   
3. ⚡ OPTIMIZE EPSILON USAGE:
   - Use lower epsilon per query (0.01-0.03 instead of 0.05-0.1)
   - Batch similar queries together
   - Use more efficient mechanisms
   
4. 🛡️  IMPROVE ERROR HANDLING:
   - Add budget checks before query submission
   - Implement query retry mechanisms
   - Better error logging and recovery
   
5. 📈 BATCH PROCESSING:
   - Group queries by type for better efficiency
   - Use holistic mechanisms for related queries
   - Reduce per-query overhead
    """)

def suggest_fixes():
    """Suggest specific code fixes"""
    
    print("\n\n🔧 SPECIFIC CODE FIXES")
    print("=" * 50)
    
    print("""
OPTION 1: Increase Privacy Budget
---------------------------------
In dp_system.py, change the budget initialization:

    # Change from:
    conn.execute(text("INSERT INTO analyst_budget VALUES ('analyst_1', 1.0, 0.0)"))
    
    # To:
    conn.execute(text("INSERT INTO analyst_budget VALUES ('analyst_1', 3.0, 0.0)"))

OPTION 2: Use Lower Epsilon Values
----------------------------------
In performance_assessment.py, modify test queries to use lower epsilon:

    # Change query submission to use lower epsilon
    # This requires modifying the QueryParser to accept epsilon parameter

OPTION 3: Add Budget Validation
-------------------------------
Add this to submit_query() method:

    def submit_query(self, query: str) -> int:
        # Check budget before submission
        with self.db_manager.get_connection() as conn:
            budget_check = conn.execute(text('''
                SELECT (epsilon_total - epsilon_spent) as remaining 
                FROM analyst_budget WHERE analyst_id = 'analyst_1'
            '''))
            remaining = budget_check.fetchone()[0]
            
            if remaining < 0.05:  # Minimum epsilon needed
                print(f"⚠️  Insufficient budget! Remaining: {remaining}")
                return None
                
        # Continue with normal query submission...
    """)

if __name__ == "__main__":
    diagnose_system()
    get_prevention_strategies()
    suggest_fixes()
