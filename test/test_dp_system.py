#!/usr/bin/env python3
"""
Test script for the new dp_system.py
"""

def test_dp_system():
    print("🚀 Testing DP System...")
    
    try:
        # Import the system
        from dp_system import DifferentialPrivacySystem
        print("✅ Successfully imported DifferentialPrivacySystem")
        
        # Create instance
        dp_system = DifferentialPrivacySystem()
        print("✅ Successfully created DifferentialPrivacySystem instance")
        
        # Test database connection
        print("🔗 Testing database connection...")
        result = dp_system.db_manager.execute_query("SELECT 1 as test")
        test_result = result.fetchone()
        print(f"✅ Database connection successful: {test_result}")
        
        # Test system initialization
        print("🏗️ Initializing system with sample data...")
        dp_system.initialize_system(num_rows=50)  # Use smaller dataset for testing
        print("✅ System initialized successfully")
        
        # Test query submission
        print("📝 Submitting test queries...")
        q1 = dp_system.submit_query("SELECT COUNT(*) FROM census_income WHERE sex='Male'")
        q2 = dp_system.submit_query("SELECT AVG(age) FROM census_income")
        q3 = dp_system.submit_query("SELECT SUM(hours_per_week) FROM census_income")
        print(f"✅ Submitted queries: {q1}, {q2}, {q3}")
        
        # Process queries
        print("⚙️ Processing queries...")
        dp_system.process_queries()
        print("✅ Queries processed")
        
        # Process batches
        print("📦 Processing batches...")
        dp_system.process_batches()
        print("✅ Batches processed")
        
        # Get results
        print("📊 Getting results...")
        results = dp_system.get_query_results()
        print(f"✅ Retrieved {len(results)} results")
        print("\n📋 Results Summary:")
        print(results[['query_id', 'query_type', 'mechanism', 'noisy_result', 'status']].to_string())
        
        print("\n🎉 All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_dp_system()
