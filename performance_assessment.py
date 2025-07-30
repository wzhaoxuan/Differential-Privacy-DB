"""
Performance Assessment Module for Differential Privacy System
Measures: Latency, CPU Utilization, Privacy Level, and Utility
"""

import time
import psutil
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sqlalchemy import text
import json
from datetime import datetime


class PerformanceAssessment:
    """Comprehensive performance assessment for differential privacy system."""
    
    def __init__(self, dp_system):
        self.dp_system = dp_system
        self.metrics = {
            'latency': [],
            'cpu_utilization': [],
            'privacy_level': [],
            'utility': []
        }
        self.query_results = []
        
    def run_performance_test(self, test_queries: List[str], num_iterations: int = 5) -> Dict[str, Any]:
        """Run comprehensive performance test with all metrics."""
        print("🚀 Starting Performance Assessment...")
        print(f"📊 Testing {len(test_queries)} queries with {num_iterations} iterations each")
        print("="*60)
        
        # Clear previous query logs for clean testing
        try:
            with self.dp_system.db_manager.get_connection() as conn:
                conn.execute(text("DELETE FROM dp_query_log"))
                conn.execute(text("UPDATE analyst_budget SET epsilon_spent = 0.0 WHERE analyst_id = 'analyst_1'"))
                conn.commit()
                print("🧹 Cleared previous query logs and reset privacy budget")
        except Exception as e:
            print(f"Warning: Could not clear previous logs: {e}")
        
        all_results = []
        
        for iteration in range(num_iterations):
            print(f"\n🔄 Iteration {iteration + 1}/{num_iterations}")
            
            # Initialize system for this iteration
            self.dp_system.initialize_system()
            
            iteration_results = []
            
            for i, query in enumerate(test_queries):
                print(f"  📝 Query {i+1}: {query[:50]}...")
                
                # Measure single query performance
                query_metrics = self._assess_single_query(query, f"test_query_{i+1}")
                query_metrics['iteration'] = iteration + 1
                query_metrics['query_id'] = i + 1
                query_metrics['query'] = query
                
                iteration_results.append(query_metrics)
                
            all_results.extend(iteration_results)
            
        # Calculate aggregate metrics
        summary = self._calculate_summary_metrics(all_results)
        
        # Generate detailed report
        report = self._generate_performance_report(all_results, summary)
        
        print("\n" + "="*60)
        print("📈 PERFORMANCE ASSESSMENT COMPLETE")
        print("="*60)
        
        return {
            'detailed_results': all_results,
            'summary': summary,
            'report': report
        }
    
    def _assess_single_query(self, query: str, query_name: str) -> Dict[str, Any]:
        """Assess performance metrics for a single query."""
        
        # 1. LATENCY MEASUREMENT
        start_time = time.time()
        cpu_start = psutil.cpu_percent(interval=None)
        
        # Execute query with DP
        query_id = self.dp_system.submit_query(query)
        self.dp_system.process_queries()
        self.dp_system.process_batches()
        
        end_time = time.time()
        cpu_end = psutil.cpu_percent(interval=None)
        
        latency = end_time - start_time
        cpu_utilization = max(cpu_end - cpu_start, 0)  # Ensure non-negative
        
        # Get results for utility and privacy assessment
        results_df = self.dp_system.get_query_results()
        print(f"  Debug: Looking for query_id {query_id} in results")
        
        # Find the current result - try to get the most recent query with this ID
        query_results = results_df[results_df['query_id'] == query_id]
        if len(query_results) == 0:
            print(f"  Warning: No results found for query_id {query_id}")
            # Use default values
            current_result = {
                'epsilon_charge': 0.05,
                'mechanism': 'Unknown',
                'noisy_result': None,
                'status': 'NOT_FOUND'
            }
        else:
            current_result = query_results.iloc[-1]  # Get the most recent result
        
        # 2. PRIVACY LEVEL MEASUREMENT
        privacy_metrics = self._assess_privacy_level(current_result)
        
        # 3. UTILITY MEASUREMENT
        utility_metrics = self._assess_utility(query, current_result)
        
        return {
            'query_name': query_name,
            'latency_seconds': latency,
            'cpu_utilization_percent': cpu_utilization,
            'privacy_epsilon_used': privacy_metrics['epsilon_used'],
            'privacy_mechanism': privacy_metrics['mechanism'],
            'privacy_budget_remaining': privacy_metrics['budget_remaining'],
            'utility_score': utility_metrics['utility_score'],
            'true_result': utility_metrics['true_result'],
            'noisy_result': utility_metrics['noisy_result'],
            'absolute_error': utility_metrics['absolute_error'],
            'relative_error_percent': utility_metrics['relative_error_percent'],
            'timestamp': datetime.now().isoformat()
        }
    
    def _assess_privacy_level(self, query_result) -> Dict[str, Any]:
        """Assess privacy level metrics."""
        
        # Get current budget status
        with self.dp_system.db_manager.get_connection() as conn:
            budget_result = conn.execute(text("""
                SELECT epsilon_total, epsilon_spent 
                FROM analyst_budget 
                WHERE analyst_id = 'analyst_1'
            """))
            budget_row = budget_result.fetchone()
            
            if budget_row:
                epsilon_total = float(budget_row[0])
                epsilon_spent = float(budget_row[1])
                budget_remaining = epsilon_total - epsilon_spent
            else:
                epsilon_total = epsilon_spent = budget_remaining = 0
        
        return {
            'epsilon_used': self._safe_get_value(query_result, 'epsilon_charge', 0.0),
            'mechanism': self._safe_get_value(query_result, 'mechanism', 'Unknown'),
            'budget_remaining': budget_remaining,
            'privacy_level': 'HIGH' if budget_remaining > 0.5 else 'MEDIUM' if budget_remaining > 0.2 else 'LOW'
        }
    
    def _safe_get_value(self, data, key, default):
        """Safely get value from pandas Series or dict."""
        try:
            if isinstance(data, dict):
                value = data.get(key, default)
            else:
                value = data[key]
            
            if pd.isna(value) or value is None:
                return default
            
            if key == 'epsilon_charge':
                return float(value)
            else:
                return str(value)
        except Exception:
            return default
    
    def _assess_utility(self, query: str, dp_result) -> Dict[str, Any]:
        """Assess utility by comparing true vs noisy results."""
        
        # Execute true query without DP
        try:
            with self.dp_system.db_manager.get_connection() as conn:
                true_result_query = conn.execute(text(query))
                true_row = true_result_query.fetchone()
                true_value = float(true_row[0]) if true_row and true_row[0] is not None else 0
        except Exception as e:
            print(f"Warning: Could not get true result for utility calculation: {e}")
            true_value = 0
        
        # Get noisy result with proper NaN handling
        try:
            noisy_result_val = self._safe_get_value(dp_result, 'noisy_result', None)
            if noisy_result_val is None or pd.isna(noisy_result_val):
                noisy_value = 0
            else:
                noisy_value = float(noisy_result_val)
        except Exception as e:
            print(f"Warning: Could not extract noisy result: {e}")
            noisy_value = 0
        
        # Calculate error metrics
        absolute_error = abs(true_value - noisy_value)
        relative_error = (absolute_error / abs(true_value)) * 100 if true_value != 0 else 0
        
        # Calculate utility score (higher is better, 0-100 scale)
        # Utility decreases as relative error increases
        if relative_error <= 5:
            utility_score = 100
        elif relative_error <= 10:
            utility_score = 90
        elif relative_error <= 20:
            utility_score = 75
        elif relative_error <= 50:
            utility_score = 50
        else:
            utility_score = max(0, 50 - (relative_error - 50))
        
        return {
            'true_result': true_value,
            'noisy_result': noisy_value,
            'absolute_error': absolute_error,
            'relative_error_percent': relative_error,
            'utility_score': utility_score
        }
    
    def _calculate_summary_metrics(self, all_results: List[Dict]) -> Dict[str, Any]:
        """Calculate summary statistics across all test results."""
        
        df = pd.DataFrame(all_results)
        
        return {
            'latency': {
                'mean_seconds': df['latency_seconds'].mean(),
                'median_seconds': df['latency_seconds'].median(),
                'std_seconds': df['latency_seconds'].std(),
                'min_seconds': df['latency_seconds'].min(),
                'max_seconds': df['latency_seconds'].max()
            },
            'cpu_utilization': {
                'mean_percent': df['cpu_utilization_percent'].mean(),
                'median_percent': df['cpu_utilization_percent'].median(),
                'std_percent': df['cpu_utilization_percent'].std(),
                'max_percent': df['cpu_utilization_percent'].max()
            },
            'privacy': {
                'total_epsilon_used': df['privacy_epsilon_used'].sum(),
                'avg_epsilon_per_query': df['privacy_epsilon_used'].mean(),
                'mechanisms_used': df['privacy_mechanism'].value_counts().to_dict(),
                'final_budget_remaining': df['privacy_budget_remaining'].iloc[-1] if len(df) > 0 else 0
            },
            'utility': {
                'mean_utility_score': df['utility_score'].mean(),
                'median_utility_score': df['utility_score'].median(),
                'std_utility_score': df['utility_score'].std(),
                'mean_relative_error_percent': df['relative_error_percent'].mean(),
                'queries_with_high_utility': len(df[df['utility_score'] >= 75]),
                'total_queries': len(df)
            },
            'overall': {
                'total_queries_tested': len(df),
                'total_test_time_seconds': df['latency_seconds'].sum(),
                'avg_queries_per_second': len(df) / df['latency_seconds'].sum() if df['latency_seconds'].sum() > 0 else 0
            }
        }
    
    def _generate_performance_report(self, results: List[Dict], summary: Dict) -> str:
        """Generate a detailed performance report."""
        
        report = []
        report.append("DIFFERENTIAL PRIVACY SYSTEM - PERFORMANCE ASSESSMENT REPORT")
        report.append("=" * 65)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Queries Tested: {summary['overall']['total_queries_tested']}")
        report.append("")
        
        # Latency Report
        report.append("🕐 LATENCY METRICS")
        report.append("-" * 20)
        report.append(f"Average Latency: {summary['latency']['mean_seconds']:.4f} seconds")
        report.append(f"Median Latency:  {summary['latency']['median_seconds']:.4f} seconds")
        report.append(f"Min Latency:     {summary['latency']['min_seconds']:.4f} seconds")
        report.append(f"Max Latency:     {summary['latency']['max_seconds']:.4f} seconds")
        report.append(f"Std Deviation:   {summary['latency']['std_seconds']:.4f} seconds")
        report.append("")
        
        # CPU Utilization Report
        report.append("💻 CPU UTILIZATION METRICS")
        report.append("-" * 27)
        report.append(f"Average CPU Usage: {summary['cpu_utilization']['mean_percent']:.2f}%")
        report.append(f"Median CPU Usage:  {summary['cpu_utilization']['median_percent']:.2f}%")
        report.append(f"Peak CPU Usage:    {summary['cpu_utilization']['max_percent']:.2f}%")
        report.append("")
        
        # Privacy Report
        report.append("🔒 PRIVACY LEVEL METRICS")
        report.append("-" * 25)
        report.append(f"Total Epsilon Used:      {summary['privacy']['total_epsilon_used']:.4f}")
        report.append(f"Average Epsilon/Query:   {summary['privacy']['avg_epsilon_per_query']:.4f}")
        report.append(f"Remaining Privacy Budget: {summary['privacy']['final_budget_remaining']:.4f}")
        report.append("Mechanisms Used:")
        for mechanism, count in summary['privacy']['mechanisms_used'].items():
            report.append(f"  - {mechanism}: {count} queries")
        report.append("")
        
        # Utility Report
        report.append("📊 UTILITY METRICS")
        report.append("-" * 18)
        report.append(f"Average Utility Score:    {summary['utility']['mean_utility_score']:.2f}/100")
        report.append(f"Median Utility Score:     {summary['utility']['median_utility_score']:.2f}/100")
        report.append(f"Average Relative Error:   {summary['utility']['mean_relative_error_percent']:.2f}%")
        report.append(f"High Utility Queries:     {summary['utility']['queries_with_high_utility']}/{summary['utility']['total_queries']} (≥75/100)")
        report.append("")
        
        # Overall Performance
        report.append("⚡ OVERALL PERFORMANCE")
        report.append("-" * 22)
        report.append(f"Total Test Duration:  {summary['overall']['total_test_time_seconds']:.2f} seconds")
        report.append(f"Throughput:          {summary['overall']['avg_queries_per_second']:.2f} queries/second")
        report.append("")
        
        # Performance Grade
        avg_utility = summary['utility']['mean_utility_score']
        avg_latency = summary['latency']['mean_seconds']
        privacy_remaining = summary['privacy']['final_budget_remaining']
        
        if avg_utility >= 80 and avg_latency < 1.0 and privacy_remaining > 0.3:
            grade = "A - EXCELLENT"
        elif avg_utility >= 70 and avg_latency < 2.0 and privacy_remaining > 0.2:
            grade = "B - GOOD"
        elif avg_utility >= 60 and avg_latency < 5.0 and privacy_remaining > 0.1:
            grade = "C - SATISFACTORY"
        else:
            grade = "D - NEEDS IMPROVEMENT"
        
        report.append(f"🏆 OVERALL PERFORMANCE GRADE: {grade}")
        report.append("=" * 65)
        
        return "\n".join(report)
    
    def save_results_to_file(self, results: Dict, filename: str = None):
        """Save performance results to JSON file."""
        if filename is None:
            filename = f"performance_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Clean results for JSON serialization
        clean_results = json.loads(json.dumps(results, default=convert_numpy))
        
        with open(filename, 'w') as f:
            json.dump(clean_results, f, indent=2)
        
        print(f"📁 Results saved to: {filename}")
        return filename


def run_comprehensive_performance_test():
    """Run a comprehensive performance test with standard queries."""
    
    # Import the DP system directly from dp_system.py
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Execute the dp_system.py file to get the classes
    with open('dp_system.py', 'r', encoding='utf-8') as f:
        exec(f.read(), globals())
    
    # Test queries covering different types
    test_queries = [
        "SELECT COUNT(*) FROM census_income",
        "SELECT AVG(age) FROM census_income",
        "SELECT SUM(`hours.per.week`) FROM census_income",
        "SELECT COUNT(*) FROM census_income WHERE sex='Female'",
        "SELECT AVG(`hours.per.week`) FROM census_income WHERE age > 30",
    ]
    
    # Initialize performance assessment
    dp_system = DifferentialPrivacySystem()
    assessor = PerformanceAssessment(dp_system)
    
    # Run comprehensive test
    results = assessor.run_performance_test(test_queries, num_iterations=3)
    
    # Print report
    print(results['report'])
    
    # Save results
    filename = assessor.save_results_to_file(results)
    
    return results, filename


if __name__ == "__main__":
    results, filename = run_comprehensive_performance_test()
