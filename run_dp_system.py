"""
Main Runner - Calls the DifferentialPrivacySystem class
"""
import time
from dp_system.differential_privacy_system import DifferentialPrivacySystem


def main():
    """Main function that calls and runs the DifferentialPrivacySystem."""
    print("🚀 Starting Differential Privacy System...")
    print("=" * 60)
    
    try:
        # Initialize the DifferentialPrivacySystem
        print("📦 Initializing DifferentialPrivacySystem...")
        dp_system = DifferentialPrivacySystem()
        
        # Initialize the system with data
        print("🔧 Setting up system and loading data...")
        dp_system.initialize_system()
        
        # Submit various queries
        print("\n📝 Submitting queries...")
        queries = [
            "SELECT COUNT(*) FROM census_income",
            "SELECT AVG(age) FROM census_income", 
            "SELECT SUM(`hours.per.week`) FROM census_income",
            "SELECT COUNT(*) FROM census_income WHERE sex='Female'"
        ]
        
        query_ids = []
        for i, query in enumerate(queries, 1):
            query_id = dp_system.submit_query(query)
            query_ids.append(query_id)
            print(f"   Query {i}: {query} → ID: {query_id}")
        
        print(f"\n✅ Submitted {len(queries)} queries: {query_ids}")
        
        # Process queries with differential privacy
        print("\n🔒 Processing queries with differential privacy...")
        dp_system.process_queries()
        
        # Process any batch queries
        print("📊 Processing batch queries...")
        time.sleep(1)  # Brief pause for processing
        dp_system.process_batches()
        
        # Get and display results
        print("\n📋 Retrieving results...")
        results = dp_system.get_query_results()
        
        print("\n🎯 DIFFERENTIAL PRIVACY RESULTS:")
        print("=" * 60)
        if not results.empty:
            # Display key columns
            display_cols = ['query_id', 'query_type', 'mechanism', 'noisy_result', 'epsilon_charge', 'status']
            print(results[display_cols].to_string(index=False))
            
            # Summary statistics
            print("\n📈 SUMMARY:")
            total_epsilon = results['epsilon_charge'].sum()
            successful_queries = len(results[results['status'] == 'DONE'])
            print(f"   • Total Queries: {len(results)}")
            print(f"   • Successful: {successful_queries}")
            print(f"   • Total ε used: {total_epsilon:.3f}")
            print(f"   • Mechanisms: {', '.join(results['mechanism'].unique())}")
        else:
            print("❌ No results found")
        
        print("=" * 60)
        print("✅ Differential Privacy System completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error running DifferentialPrivacySystem: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
