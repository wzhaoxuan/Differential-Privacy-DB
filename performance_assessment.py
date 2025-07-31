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

# Import the shared queries from dp_system
from dp_system import DifferentialPrivacySystem, DEFAULT_TEST_QUERIES


class PerformanceAssessment:
    """Comprehensive performance assessment for differential privacy system."""
    
    def __init__(self, dp_system):
        self.dp_system = dp_system
        # Use the exact same queries from dp_system.py
        self.test_queries = DEFAULT_TEST_QUERIES.copy()
        self.metrics = {
            'latency': [],
            'cpu_utilization': [],
            'privacy_level': [],
            'utility': []
        }
        self.query_results = []
        
        print(f"📊 Performance Assessment initialized with {len(self.test_queries)} queries from dp_system.py")
        for i, query in enumerate(self.test_queries, 1):
            print(f"   {i}. {query}")
        
    def run_performance_test(self, num_iterations: int = 5) -> Dict[str, Any]:
        """Run performance test using the same queries as dp_system.py."""
        print(f"\n🚀 Running Performance Assessment")
        print(f"🔄 Running {num_iterations} iterations for statistical accuracy...")
        
        # Clear previous query logs for clean testing
        self._cleanup_previous_results()
        
        iteration_results = []
        
        for iteration in range(num_iterations):
            print(f"\n--- Iteration {iteration + 1}/{num_iterations} ---")
            
            # Initialize fresh system for each iteration
            test_system = DifferentialPrivacySystem()
            test_system.initialize_system()
            
            # Track query IDs for this iteration
            start_time = time.time()
            
            # Submit queries and track their IDs
            iteration_query_ids = []
            for query in self.test_queries:
                query_id = test_system.submit_query(query)
                iteration_query_ids.append(query_id)
            
            print(f"Submitted queries: {', '.join(map(str, iteration_query_ids))}")
            
            # Process with DP
            test_system.process_queries()
            test_system.process_batches()
            
            # Get results for ONLY this iteration's queries
            results = self._get_iteration_specific_results(test_system, iteration_query_ids)
            end_time = time.time()
            
            print(f"📊 Iteration {iteration + 1} Results ({len(results)} queries):")
            if not results.empty:
                print(results[['query_id', 'query_type', 'mechanism', 'noisy_result', 'epsilon_charge', 'status']].to_string(index=False))
            else:
                print("No results found")
            
            # Analyze results
            iteration_result = self._analyze_iteration_results(
                results, iteration + 1, start_time, end_time
            )
            iteration_results.append(iteration_result)
        
        # Compile final results
        return self._compile_final_results(iteration_results)
    
    def _cleanup_previous_results(self):
        """Clear previous query logs for clean testing."""
        try:
            with self.dp_system.db_manager.get_connection() as conn:
                # Clear ALL previous query logs
                conn.execute(text("DELETE FROM dp_query_log"))
                conn.execute(text("DELETE FROM dp_batch"))
                conn.execute(text("DELETE FROM dp_batch_member"))
                conn.execute(text("DELETE FROM dp_measurement"))
                
                # Reset privacy budget
                conn.execute(text("UPDATE analyst_budget SET epsilon_spent = 0.0 WHERE analyst_id = 'analyst_1'"))
                conn.commit()
                
                print("🧹 Cleared all previous query logs and reset privacy budget")
        except Exception as e:
            print(f"Warning: Could not clear previous logs: {e}")
    
    def _get_iteration_specific_results(self, test_system, query_ids: List[int]) -> pd.DataFrame:
        """Get results for specific query IDs only."""
        if not query_ids:
            return pd.DataFrame()
        
        query_ids_str = ','.join(map(str, query_ids))
        with test_system.db_manager.get_connection() as conn:
            return pd.read_sql(
                f"SELECT * FROM dp_query_log WHERE query_id IN ({query_ids_str}) ORDER BY query_id", 
                conn
            )
    
    def _analyze_iteration_results(self, results_df: pd.DataFrame, iteration: int, start_time: float, end_time: float) -> Dict[str, Any]:
        """Analyze results from a single iteration."""
        total_latency = end_time - start_time
        
        # Process each query result
        query_results = []
        for _, row in results_df.iterrows():
            query_result = {
                'iteration': iteration,
                'query_id': row['query_id'],
                'query_text': row['raw_sql'],
                'latency': total_latency / len(results_df) if len(results_df) > 0 else total_latency,
                'cpu_usage': psutil.cpu_percent(),
                'utility_score': self._calculate_utility_score(row),
                'epsilon_used': row['epsilon_charge'],
                'mechanism': row['mechanism'],
                'noisy_result': row['noisy_result'],
                'status': row['status']
            }
            query_results.append(query_result)
        
        return {
            'iteration': iteration,
            'total_latency': total_latency,
            'query_results': query_results,
            'summary': {
                'avg_latency': total_latency / len(results_df) if len(results_df) > 0 else 0,
                'total_epsilon': results_df['epsilon_charge'].sum() if not results_df.empty else 0,
                'success_rate': len(results_df[results_df['status'] == 'DONE']) / len(results_df) * 100 if not results_df.empty else 0
            }
        }
    
    def _calculate_utility_score(self, query_result) -> float:
        """Calculate utility score for a query result."""
        try:
            # Get true result
            with self.dp_system.db_manager.get_connection() as conn:
                true_result = conn.execute(text(query_result['raw_sql']))
                true_row = true_result.fetchone()
                true_value = float(true_row[0]) if true_row and true_row[0] is not None else 0
            
            # Get noisy result
            noisy_value = float(query_result['noisy_result']) if pd.notna(query_result['noisy_result']) else 0
            
            # Calculate relative error
            if true_value != 0:
                relative_error = abs(true_value - noisy_value) / abs(true_value) * 100
            else:
                relative_error = 0 if noisy_value == 0 else 100
            
            # Convert to utility score (0-100, higher is better)
            if relative_error <= 5:
                return 100
            elif relative_error <= 10:
                return 90
            elif relative_error <= 20:
                return 75
            elif relative_error <= 50:
                return 50
            else:
                return max(0, 50 - (relative_error - 50))
        except Exception:
            return 0
    
    def _compile_final_results(self, iteration_results: List[Dict]) -> Dict[str, Any]:
        """Compile final results from all iterations."""
        all_query_results = []
        for iteration_result in iteration_results:
            all_query_results.extend(iteration_result['query_results'])
        
        # Calculate summary statistics
        if all_query_results:
            df = pd.DataFrame(all_query_results)
            
            summary = {
                'latency': {
                    'average': df['latency'].mean(),
                    'median': df['latency'].median(),
                    'std': df['latency'].std(),
                    'min': df['latency'].min(),
                    'max': df['latency'].max()
                },
                'cpu': {
                    'average': df['cpu_usage'].mean(),
                    'median': df['cpu_usage'].median(),
                    'max': df['cpu_usage'].max()
                },
                'utility': {
                    'average': df['utility_score'].mean(),
                    'median': df['utility_score'].median(),
                    'std': df['utility_score'].std()
                },
                'privacy': {
                    'total_epsilon_used': df['epsilon_used'].sum(),
                    'avg_epsilon_per_query': df['epsilon_used'].mean(),
                    'mechanisms_used': df['mechanism'].value_counts().to_dict()
                },
                'overall': {
                    'total_queries': len(df),
                    'success_rate': len(df[df['status'] == 'DONE']) / len(df) * 100,
                    'throughput': len(df) / df['latency'].sum() if df['latency'].sum() > 0 else 0
                }
            }
        else:
            summary = {}
        
        return {
            'query_results': all_query_results,
            'summary': summary,
            'test_queries': self.test_queries,
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_performance_report(self, results: List[Dict], summary: Dict) -> str:
        """Generate a detailed performance report."""
        
        report = []
        report.append("DIFFERENTIAL PRIVACY SYSTEM - PERFORMANCE ASSESSMENT REPORT")
        report.append("=" * 65)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Test Queries from Differential Privacy System: {len(self.test_queries)}")
        for i, query in enumerate(self.test_queries, 1):
            report.append(f"  {i}. {query}")
        report.append("")
        
        if summary:
            # Latency Report
            report.append("🕐 LATENCY METRICS")
            report.append("-" * 20)
            report.append(f"Average Latency: {summary['latency']['average']:.4f} seconds")
            report.append(f"Median Latency:  {summary['latency']['median']:.4f} seconds")
            report.append(f"Min Latency:     {summary['latency']['min']:.4f} seconds")
            report.append(f"Max Latency:     {summary['latency']['max']:.4f} seconds")
            report.append("")
            
            # CPU Utilization Report
            report.append("💻 CPU UTILIZATION METRICS")
            report.append("-" * 27)
            report.append(f"Average CPU Usage: {summary['cpu']['average']:.2f}%")
            report.append(f"Median CPU Usage:  {summary['cpu']['median']:.2f}%")
            report.append(f"Peak CPU Usage:    {summary['cpu']['max']:.2f}%")
            report.append("")
            
            # Privacy Report
            report.append("🔒 PRIVACY LEVEL METRICS")
            report.append("-" * 25)
            report.append(f"Total Epsilon Used:      {summary['privacy']['total_epsilon_used']:.4f}")
            report.append(f"Average Epsilon/Query:   {summary['privacy']['avg_epsilon_per_query']:.4f}")
            report.append("Mechanisms Used:")
            for mechanism, count in summary['privacy']['mechanisms_used'].items():
                report.append(f"  - {mechanism}: {count} queries")
            report.append("")
            
            # Utility Report
            report.append("📊 UTILITY METRICS")
            report.append("-" * 18)
            report.append(f"Average Utility Score:    {summary['utility']['average']:.2f}/100")
            report.append(f"Median Utility Score:     {summary['utility']['median']:.2f}/100")
            report.append("")
            
            # Overall Performance
            report.append("⚡ OVERALL PERFORMANCE")
            report.append("-" * 22)
            report.append(f"Success Rate:        {summary['overall']['success_rate']:.1f}%")
            report.append(f"Throughput:          {summary['overall']['throughput']:.2f} queries/second")
            report.append("")
        
        report.append("=" * 65)
        
        return "\n".join(report)


def run_comprehensive_performance_test():
    """Run comprehensive performance test using dp_system.py queries."""
    print("🎯 Comprehensive Performance Assessment")
    print("=" * 60)
    
    # Create DP system
    dp_system = DifferentialPrivacySystem()
    
    # Create performance assessor
    assessor = PerformanceAssessment(dp_system)
    
    # Run performance test with the same queries from dp_system.py
    results = assessor.run_performance_test(num_iterations=3)
    
    # Generate report
    report = assessor._generate_performance_report(
        results.get('query_results', []), 
        results.get('summary', {})
    )
    
    print(report)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"performance_results_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {filename}")
    return results, filename


if __name__ == "__main__":
    results, filename = run_comprehensive_performance_test()
