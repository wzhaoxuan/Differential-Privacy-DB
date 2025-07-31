"""
Quick Performance Test for DP System
Tests: Latency, CPU, Privacy, and Utility using the existing performance_assessment.py
"""

import time
from dp_system import DifferentialPrivacySystem
from performance_assessment import PerformanceAssessment


def quick_performance_test():
    """Run a quick performance assessment using the existing PerformanceAssessment class."""
    print("🚀 Quick Performance Test for Differential Privacy System")
    print("=" * 60)
    
    # Initialize system
    print("1️⃣  Initializing system...")
    dp_system = DifferentialPrivacySystem()
    
    # Create performance assessor
    assessor = PerformanceAssessment(dp_system)
    
    # Test queries - using correct column names from working dp_system.py
    queries = [
        "SELECT COUNT(*) FROM census_income",
        "SELECT AVG(age) FROM census_income", 
        "SELECT SUM(`hours.per.week`) FROM census_income",  # Correct column name with backticks
        "SELECT COUNT(*) FROM census_income WHERE `marital.status`='Divorced'"
    ]
    
    print(f"\n2️⃣  Running performance test with {len(queries)} queries...")
    
    # Run performance assessment with 3 iterations for speed
    try:
        results = assessor.run_performance_test(queries, num_iterations=3)
        
        print("\n🎯 PERFORMANCE RESULTS")
        print("=" * 40)
        
        # Display summary from the assessment
        if 'summary' in results:
            summary = results['summary']
            
            # Latency
            if 'latency' in summary:
                lat = summary['latency']
                print(f"⏱️  Latency:")
                print(f"   Average: {lat.get('mean_seconds', 0):.3f} seconds")
                print(f"   Range: {lat.get('min_seconds', 0):.3f} - {lat.get('max_seconds', 0):.3f} seconds")
            
            # CPU Utilization  
            if 'cpu_utilization' in summary:
                cpu = summary['cpu_utilization']
                print(f"💻 CPU Usage:")
                print(f"   Average: {cpu.get('mean_percent', 0):.1f}%")
                print(f"   Peak: {cpu.get('max_percent', 0):.1f}%")
            
            # Privacy Level
            if 'privacy' in summary:
                priv = summary['privacy']
                print(f"🔒 Privacy:")
                print(f"   Epsilon Used: {priv.get('total_epsilon_used', 0):.4f}")
                print(f"   Budget Remaining: {priv.get('final_budget_remaining', 0):.4f}")
            
            # Utility
            if 'utility' in summary:
                util = summary['utility']
                print(f"📊 Utility:")
                print(f"   Average Score: {util.get('mean_utility_score', 0):.1f}/100")
                print(f"   Average Error: {util.get('mean_relative_error_percent', 0):.1f}%")
        
        print("=" * 40)
        print("✅ Performance assessment completed successfully!")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during performance assessment: {e}")
        print("\nTrying fallback with simpler queries...")
        
        # Fallback with simpler queries
        simple_queries = [
            "SELECT COUNT(*) FROM census_income",
            "SELECT AVG(age) FROM census_income"
        ]
        
        try:
            results = assessor.run_performance_test(simple_queries, num_iterations=2)
            print("✅ Fallback assessment completed!")
            return results
        except Exception as e2:
            print(f"❌ Fallback also failed: {e2}")
            return None


if __name__ == "__main__":
    quick_performance_test()
