"""
Quick Performance Test for DP System
Tests: Latency, CPU, Privacy, and Utility
"""

import time
import psutil
from dp_system import DifferentialPrivacySystem


def quick_performance_test():
    """Run a quick performance assessment."""
    print("🚀 Quick Performance Test for Differential Privacy System")
    print("=" * 60)
    
    # Initialize system
    print("1️⃣  Initializing system...")
    dp_system = DifferentialPrivacySystem()
    dp_system.initialize_system(num_rows=100)
    
    # Test queries
    queries = [
        "SELECT COUNT(*) FROM census_income",
        "SELECT AVG(age) FROM census_income", 
        "SELECT SUM(hours_per_week) FROM census_income",
        "SELECT COUNT(*) FROM census_income WHERE `marital.status`='Divorced'"
    ]
    
    print("\n2️⃣  Measuring Performance Metrics...")
    
    # Initialize system
    dp_system = DifferentialPrivacySystem()
    assessor = PerformanceAssessment(dp_system)
    
    # Quick test queries
    test_queries = [
        "SELECT COUNT(*) FROM census_income",
        "SELECT AVG(age) FROM census_income", 
        "SELECT SUM(hours_per_week) FROM census_income"
    ]
    
    print(f"Testing {len(test_queries)} queries...")
    
    # Run performance test with 2 iterations
    results = assessor.run_performance_test(test_queries, num_iterations=2)
    
    # Display summary
    summary = results['summary']
    
    print("\n📈 PERFORMANCE SUMMARY")
    print("-" * 30)
    print(f"⏱️  Average Latency: {summary['latency']['mean_seconds']:.3f}s")
    print(f"💻 Average CPU Usage: {summary['cpu_utilization']['mean_percent']:.1f}%")
    print(f"🔒 Privacy Budget Used: {summary['privacy']['total_epsilon_used']:.3f}")
    print(f"📊 Average Utility Score: {summary['utility']['mean_utility_score']:.1f}/100")
    print(f"📉 Average Error: {summary['utility']['mean_relative_error_percent']:.1f}%")
    
    # Quick performance assessment
    latency_score = "FAST" if summary['latency']['mean_seconds'] < 1.0 else "MEDIUM" if summary['latency']['mean_seconds'] < 3.0 else "SLOW"
    cpu_score = "LOW" if summary['cpu_utilization']['mean_percent'] < 20 else "MEDIUM" if summary['cpu_utilization']['mean_percent'] < 50 else "HIGH"
    privacy_score = "GOOD" if summary['privacy']['final_budget_remaining'] > 0.5 else "MEDIUM" if summary['privacy']['final_budget_remaining'] > 0.2 else "LOW"
    utility_score = "HIGH" if summary['utility']['mean_utility_score'] >= 80 else "MEDIUM" if summary['utility']['mean_utility_score'] >= 60 else "LOW"
    
    print(f"\n🎯 QUICK ASSESSMENT:")
    print(f"   Latency: {latency_score}")
    print(f"   CPU Usage: {cpu_score}")
    print(f"   Privacy Level: {privacy_score}")
    print(f"   Utility: {utility_score}")
    
    return results


if __name__ == "__main__":
    quick_performance_test()
