#!/usr/bin/env python3

import mysql.connector
import pandas as pd

# Quick diagnosis of DP system failures
def quick_check():
    print("🔍 QUICK DIAGNOSIS OF DP SYSTEM FAILURES")
    print("=" * 50)
    
    # Connect directly to MySQL
    connection = mysql.connector.connect(
        host='localhost',
        user='root',
        password='Wx991230@',
        database='adult_income_dp'
    )
    
    cursor = connection.cursor()
    
    # 1. Check privacy budget
    print("\n💰 PRIVACY BUDGET:")
    cursor.execute("SELECT analyst_id, epsilon_total, epsilon_spent, (epsilon_total - epsilon_spent) as remaining FROM analyst_budget")
    for row in cursor.fetchall():
        print(f"  Analyst: {row[0]} | Total: {row[1]} | Spent: {row[2]} | Remaining: {row[3]}")
    
    # 2. Query status summary
    print("\n📊 QUERY STATUS SUMMARY:")
    cursor.execute("SELECT status, COUNT(*) as count FROM dp_query_log GROUP BY status")
    for row in cursor.fetchall():
        print(f"  Status: {row[0]} | Count: {row[1]}")
    
    # 3. Mechanism summary
    print("\n🔧 MECHANISM SUMMARY:")
    cursor.execute("SELECT mechanism, COUNT(*) as count FROM dp_query_log GROUP BY mechanism")
    for row in cursor.fetchall():
        print(f"  Mechanism: {row[0]} | Count: {row[1]}")
    
    # 4. Recent failed queries
    print("\n❌ SAMPLE FAILED QUERIES:")
    cursor.execute("SELECT query_id, status, mechanism FROM dp_query_log WHERE status = 'FAILED' OR mechanism IS NULL LIMIT 5")
    for row in cursor.fetchall():
        print(f"  Query {row[0]}: Status={row[1]}, Mechanism={row[2]}")
    
    # 5. Budget exhaustion check
    cursor.execute("SELECT (epsilon_total - epsilon_spent) as remaining FROM analyst_budget WHERE analyst_id = 'analyst_1'")
    remaining = cursor.fetchone()[0]
    
    print(f"\n⚠️  DIAGNOSIS:")
    if remaining <= 0:
        print("  🚨 PRIVACY BUDGET EXHAUSTED!")
        print("  📊 This is why queries are failing.")
        print("  💡 Solution: Reset budget or increase total budget.")
    elif remaining < 0.05:
        print("  ⚠️  PRIVACY BUDGET VERY LOW!")
        print("  📊 Queries need minimum 0.05 epsilon.")
        print("  💡 Solution: Reset budget or use lower epsilon values.")
    else:
        print("  ✅ Privacy budget seems adequate.")
        print("  📊 Failures may be due to other issues.")
    
    connection.close()

if __name__ == "__main__":
    quick_check()
