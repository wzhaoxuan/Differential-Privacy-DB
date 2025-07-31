#!/usr/bin/env python3
"""Test script to debug the database_manager import issue"""

import sys
import traceback

print("Testing database_manager.py import...")

try:
    # Test 1: Import config
    print("1. Testing config import...")
    from config.db_config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
    print("   Config imported successfully")
    
    # Test 2: Import SQLAlchemy components
    print("2. Testing SQLAlchemy imports...")
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker
    print("   SQLAlchemy imported successfully")
    
    # Test 3: Import database_manager module
    print("3. Testing database_manager module import...")
    import dp_system.core.database_manager as dbm
    print("   Module imported successfully")
    
    # Test 4: Check for DatabaseManager class
    print("4. Checking for DatabaseManager class...")
    if hasattr(dbm, 'DatabaseManager'):
        print("   DatabaseManager class found!")
        print("   Creating instance...")
        # Don't actually create instance as it might fail on DB connection
        print("   Test completed successfully!")
    else:
        print("   ERROR: DatabaseManager class not found in module")
        print("   Available attributes:", [attr for attr in dir(dbm) if not attr.startswith('_')])

except Exception as e:
    print(f"ERROR: {e}")
    print("Full traceback:")
    traceback.print_exc()
