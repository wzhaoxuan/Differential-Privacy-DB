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
        
        print(f"Performance Assessment initialized with {len(self.test_queries)} queries from dp_system.py")
        for i, query in enumerate(self.test_queries, 1):
            print(f"   {i}. {query}")
        
    def run_performance_test(self, num_iterations: int) -> Dict[str, Any]:
        """Run performance test using the same queries as dp_system.py."""
        print(f"\nRunning Performance Assessment")
        print(f"Running {num_iterations} iterations for statistical accuracy...")
        
        # Clear previous query logs for clean testing
        self._cleanup_previous_results()
        
        iteration_results = []
        
        for iteration in range(num_iterations):
            print(f"\n--- Iteration {iteration + 1}/{num_iterations} ---")
            
            # Initialize fresh system for each iteration
            test_system = DifferentialPrivacySystem()
            test_system.initialize_system()

            # Reset budget for this iteration
            self._reset_analyst_budget(test_system, 'analyst_1')
            
            # Track query IDs for this iteration
            start_time = time.time()
            
            # Submit queries and track their IDs
            iteration_query_ids = []
            for query in self.test_queries:
                query_id = test_system.submit_query(query, 'analyst_1')
                iteration_query_ids.append(query_id)
            
            print(f"Submitted queries: {', '.join(map(str, iteration_query_ids))}")
            
            # Process with DP
            test_system.process_queries('analyst_1')
            test_system.process_batches()
            
            # Get results for ONLY this iteration's queries
            results = self._get_iteration_specific_results(test_system, iteration_query_ids)
            end_time = time.time()
            
            print(f"Iteration {iteration + 1} Results ({len(results)} queries):")
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
    
    def _reset_analyst_budget(self, test_system, analyst_id: str, total_budget: float = 1.0):
        """Reset budget for analyst to fresh state."""
        try:
            with test_system.db_manager.get_connection() as conn:
                conn.execute(text("""
                    UPDATE analyst_budget 
                    SET epsilon_spent = 0.0, epsilon_total = :epsilon_total 
                    WHERE analyst_id = :analyst_id
                """), {
                    'analyst_id': analyst_id,
                    'epsilon_total': total_budget
                })
                conn.commit()
                print(f"Reset budget to {total_budget} epsilon for {analyst_id}")
        except Exception as e:
            print(f"Warning: Could not reset budget for {analyst_id}: {e}")

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
        """Analyze results from a single iteration with per-query measurements."""
        
        # Process each query result
        query_results = []
        print(f"\n🔍 Detailed Per-Query Performance Analysis for Iteration {iteration}:")
        print("-" * 70)
        
        for _, row in results_df.iterrows():
            # Measure individual query latency and CPU
            query_start_time = time.time()
            cpu_before = psutil.cpu_percent(interval=None)  # Get current CPU usage
            
            # Calculate accuracy loss (this involves executing the true query)
            accuracy_loss = self._calculate_utility_score(row)
            true_result = self._get_true_result(row['raw_sql'])
            
            # Measure after processing
            query_end_time = time.time()
            cpu_after = psutil.cpu_percent(interval=None)
            
            # Calculate individual metrics
            individual_latency = query_end_time - query_start_time
            individual_cpu = max(cpu_after, cpu_before)  # Take the higher reading
            
            # Display individual query performance
            query_summary = row['raw_sql'][:50] + "..." if len(row['raw_sql']) > 50 else row['raw_sql']
            print(f"Query {row['query_id']} ({row['query_type']}):")
            print(f"  Latency:        {individual_latency:.4f} seconds")
            print(f"  CPU Usage:      {individual_cpu:.1f}%")
            print(f"  Accuracy Loss:  {accuracy_loss:.2f}%")
            print(f"  Mechanism:      {row['mechanism']}, Epsilon: {row['epsilon_charge']}")
            print(f"  SQL:            {query_summary}")
            print(f"  Results:        True: {true_result:.4f}, Noisy: {row['noisy_result']:.4f}")
            print()
            
            query_result = {
                'iteration': iteration,
                'query_id': row['query_id'],
                'query_text': row['raw_sql'],
                'query_type': row['query_type'],
                'latency': individual_latency,              # Individual query latency
                'cpu_usage': individual_cpu,                # Individual query CPU usage
                'accuracy_loss_percent': accuracy_loss,
                'epsilon_used': row['epsilon_charge'],
                'mechanism': row['mechanism'],
                'noisy_result': row['noisy_result'],
                'true_result': true_result,
                'status': row['status']
            }
            query_results.append(query_result)
        
        return {
            'iteration': iteration,
            'total_latency': end_time - start_time,         # Total batch time for reference
            'query_results': query_results,                 # Individual measurements
            'summary': {
                'avg_latency': sum(q['latency'] for q in query_results) / len(query_results) if query_results else 0,
                'avg_cpu': sum(q['cpu_usage'] for q in query_results) / len(query_results) if query_results else 0,
                'total_epsilon': results_df['epsilon_charge'].sum() if not results_df.empty else 0,
                'success_rate': len(results_df[results_df['status'] == 'DONE']) / len(results_df) * 100 if not results_df.empty else 0
            }
        }

    def _get_true_result(self, raw_sql: str) -> float:
        """Get the true result for a query."""
        try:
            with self.dp_system.db_manager.get_connection() as conn:
                result = conn.execute(text(raw_sql))
                row = result.fetchone()
                return float(row[0]) if row and row[0] is not None else 0
        except Exception as e:
            print(f"Warning: Could not get true result for query: {e}")
            return 0
    
    def _calculate_utility_score(self, query_result) -> float:
        """Calculate accuracy loss (%) for a query result by comparing noisy vs true result."""
        try:
            # Get true result by executing the raw SQL
            true_value = self._get_true_result(query_result['raw_sql'])
            
            # Get noisy result
            noisy_value = float(query_result['noisy_result']) if pd.notna(query_result['noisy_result']) else 0
            
            # Calculate accuracy loss percentage
            if true_value != 0:
                accuracy_loss = abs(true_value - noisy_value) / abs(true_value) * 100
            else:
                # Handle case where true value is 0
                accuracy_loss = abs(noisy_value) if noisy_value != 0 else 0

            return accuracy_loss
        except Exception as e:
            print(f"Warning: Could not calculate accuracy loss for query {query_result.get('query_id', 'unknown')}: {e}")
            return float('inf')
    
    def _compile_final_results(self, iteration_results: List[Dict]) -> Dict[str, Any]:
        """Compile final results from all iterations with per-query metrics."""
        all_query_results = []
        for iteration_result in iteration_results:
            all_query_results.extend(iteration_result['query_results'])
        
        # Calculate summary statistics
        if all_query_results:
            df = pd.DataFrame(all_query_results)
            
            # Filter out infinite values for accuracy loss calculations
            valid_accuracy_loss = df[df['accuracy_loss_percent'] != float('inf')]['accuracy_loss_percent']
            
            # Calculate per-iteration epsilon usage
            num_iterations = len(iteration_results)
            epsilon_per_iteration = df['epsilon_used'].sum() / num_iterations if num_iterations > 0 else 0

            summary = {
                'latency': {  # Now based on individual query measurements
                    'average': df['latency'].mean(),
                    'median': df['latency'].median(),
                    'std': df['latency'].std(),
                    'min': df['latency'].min(),
                    'max': df['latency'].max()
                },
                'cpu': {  # Now based on individual query measurements
                    'average': df['cpu_usage'].mean(),
                    'median': df['cpu_usage'].median(),
                    'std': df['cpu_usage'].std(),
                    'min': df['cpu_usage'].min(),
                    'max': df['cpu_usage'].max()
                },
                'utility': {  # Now represents accuracy loss
                    'avg_accuracy_loss_percent': valid_accuracy_loss.mean() if not valid_accuracy_loss.empty else 0,
                    'median_accuracy_loss_percent': valid_accuracy_loss.median() if not valid_accuracy_loss.empty else 0,
                    'min_accuracy_loss_percent': valid_accuracy_loss.min() if not valid_accuracy_loss.empty else 0,
                    'max_accuracy_loss_percent': valid_accuracy_loss.max() if not valid_accuracy_loss.empty else 0,
                    'std_accuracy_loss_percent': valid_accuracy_loss.std() if not valid_accuracy_loss.empty else 0
                },
                'privacy': {
                    'epsilon_per_iteration': epsilon_per_iteration, 
                    'total_iterations': num_iterations,  
                    'total_epsilon_across_iterations': df['epsilon_used'].sum(),
                    'avg_epsilon_per_query': df['epsilon_used'].mean(),
                    'mechanisms_used': df['mechanism'].value_counts().to_dict()
                },
                'overall': {
                    'total_queries': len(df),
                    'queries_per_iteration': len(df) // num_iterations if num_iterations > 0 else 0,
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
        """Generate a detailed performance report with per-query metrics."""
        
        report = []
        report.append("DIFFERENTIAL PRIVACY SYSTEM - PERFORMANCE ASSESSMENT REPORT")
        report.append("=" * 65)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Test Queries from Differential Privacy System: {len(self.test_queries)}")
        report.append("")
        
        # Add detailed per-query performance analysis
        if results:
            report.append("🔍 DETAILED PER-QUERY PERFORMANCE ANALYSIS")
            report.append("=" * 48)
            
            # Group results by actual SQL query (not query type)
            df = pd.DataFrame(results)
            
            # Group by the actual SQL text to show performance for each unique query
            for i, test_query in enumerate(self.test_queries, 1):
                # Find all results for this specific SQL query
                query_results = df[df['query_text'] == test_query]
                
                if not query_results.empty:
                    # Get first row to determine query type and mechanism
                    first_result = query_results.iloc[0]
                    
                    report.append(f"\n📊 QUERY {i} ({first_result['query_type'].upper()}):")
                    report.append("-" * 30)
                    query_preview = test_query[:80] + "..." if len(test_query) > 80 else test_query
                    report.append(f"SQL: {query_preview}")
                    report.append("")
                    
                    # Show results from each iteration for this query
                    valid_results = query_results[query_results['accuracy_loss_percent'] != float('inf')]
                    
                    if not valid_results.empty:
                        # Calculate statistics for this specific query across iterations
                        avg_latency = valid_results['latency'].mean()
                        avg_cpu = valid_results['cpu_usage'].mean()
                        avg_loss = valid_results['accuracy_loss_percent'].mean()
                        median_latency = valid_results['latency'].median()
                        median_cpu = valid_results['cpu_usage'].median()
                        median_loss = valid_results['accuracy_loss_percent'].median()
                        std_latency = valid_results['latency'].std()
                        std_cpu = valid_results['cpu_usage'].std()
                        std_loss = valid_results['accuracy_loss_percent'].std()
                        
                        # Performance summary for this specific query
                        report.append(f"Performance Across {len(valid_results)} Iterations:")
                        report.append(f"  ⏱️  Latency:        Avg: {avg_latency:.4f}s, Median: {median_latency:.4f}s, Std: {std_latency:.4f}s")
                        report.append(f"  💻 CPU Usage:      Avg: {avg_cpu:.1f}%, Median: {median_cpu:.1f}%, Std: {std_cpu:.1f}%")
                        report.append(f"  📊 Accuracy Loss:  Avg: {avg_loss:.2f}%, Median: {median_loss:.2f}%, Std: {std_loss:.2f}%")
                        report.append(f"  🔒 Mechanism:      {first_result['mechanism']}")
                        report.append(f"  🛡️  Avg Epsilon:    {valid_results['epsilon_used'].mean():.3f}")
                        report.append("")
                        
                        # Show individual iteration results for this query
                        report.append("Individual Iteration Results:")
                        for _, row in valid_results.iterrows():
                            report.append(f"  Iteration {row['iteration']}: "
                                        f"Latency: {row['latency']:.4f}s, "
                                        f"CPU: {row['cpu_usage']:.1f}%, "
                                        f"Loss: {row['accuracy_loss_percent']:.2f}%, "
                                        f"True: {row['true_result']:.2f}, "
                                        f"Noisy: {row['noisy_result']:.2f}")
                        report.append("")
        
        report.append("=" * 65)
        report.append("")
        
        # Overall summary by query type (keep this for comparison)
        report.append("📈 SUMMARY BY QUERY TYPE:")
        report.append("-" * 28)
        
        for query_type in ['single', 'mean', 'batch']:
            type_results = df[df['query_type'] == query_type]
            valid_results = type_results[type_results['accuracy_loss_percent'] != float('inf')]
            
            if not valid_results.empty:
                avg_latency = valid_results['latency'].mean()
                avg_cpu = valid_results['cpu_usage'].mean()
                avg_loss = valid_results['accuracy_loss_percent'].mean()
                
                report.append(f"{query_type.upper()} Queries ({len(valid_results)} measurements):")
                report.append(f"  Avg Latency: {avg_latency:.4f}s, Avg CPU: {avg_cpu:.1f}%, Avg Loss: {avg_loss:.2f}%")
        
        report.append("")
        report.append("=" * 65)
        report.append("")
    
        if summary:
            # Latency Report - Now based on individual measurements
            report.append("🕐 OVERALL LATENCY METRICS (Per-Query Measurements)")
            report.append("-" * 54)
            report.append(f"Average Latency: {summary['latency']['average']:.4f} seconds")
            report.append(f"Median Latency:  {summary['latency']['median']:.4f} seconds")
            report.append(f"Min Latency:     {summary['latency']['min']:.4f} seconds")
            report.append(f"Max Latency:     {summary['latency']['max']:.4f} seconds")
            report.append(f"Std Dev Latency: {summary['latency']['std']:.4f} seconds")
            report.append("")
            
            # CPU Utilization Report - Now based on individual measurements
            report.append("💻 OVERALL CPU UTILIZATION METRICS (Per-Query Measurements)")
            report.append("-" * 61)
            report.append(f"Average CPU Usage: {summary['cpu']['average']:.2f}%")
            report.append(f"Median CPU Usage:  {summary['cpu']['median']:.2f}%")
            report.append(f"Min CPU Usage:     {summary['cpu']['min']:.2f}%")
            report.append(f"Peak CPU Usage:    {summary['cpu']['max']:.2f}%")
            report.append(f"Std Dev CPU Usage: {summary['cpu']['std']:.2f}%")
            report.append("")
            
            # Privacy Report
            report.append("🔒 PRIVACY LEVEL METRICS")
            report.append("-" * 25)
            report.append(f"Epsilon Used Per Iteration:  {summary['privacy']['epsilon_per_iteration']:.4f}")
            report.append(f"Number of Iterations:        {summary['privacy']['total_iterations']}")
            report.append(f"Total Budget Per Iteration:  1.0000 (reset each iteration)")
            report.append(f"Budget Utilization:          {summary['privacy']['epsilon_per_iteration']/1.0*100:.1f}%")
            report.append(f"Average Epsilon/Query:       {summary['privacy']['avg_epsilon_per_query']:.4f}")
            report.append("")
            report.append("Mechanisms Used:")
            for mechanism, count in summary['privacy']['mechanisms_used'].items():
                report.append(f"  - {mechanism}: {count} queries")
            report.append("")
            
            # Utility Report - UPDATED FOR ACCURACY LOSS
            report.append("📊 OVERALL UTILITY METRICS (Accuracy Loss)")
            report.append("-" * 42)
            report.append(f"Average Accuracy Loss:    {summary['utility']['avg_accuracy_loss_percent']:.2f}%")
            report.append(f"Median Accuracy Loss:     {summary['utility']['median_accuracy_loss_percent']:.2f}%")
            report.append(f"Min Accuracy Loss:        {summary['utility']['min_accuracy_loss_percent']:.2f}%")
            report.append(f"Max Accuracy Loss:        {summary['utility']['max_accuracy_loss_percent']:.2f}%")
            report.append(f"Std Dev Accuracy Loss:    {summary['utility']['std_accuracy_loss_percent']:.2f}%")
            report.append("")
            
            # Overall Performance
            report.append("⚡ OVERALL PERFORMANCE")
            report.append("-" * 22)
            report.append(f"Success Rate:        {summary['overall']['success_rate']:.1f}%")
            report.append(f"Queries Per Iteration: {summary['overall']['queries_per_iteration']}")
            report.append(f"Total Iterations:    {summary['privacy']['total_iterations']}")
            report.append(f"Throughput:          {summary['overall']['throughput']:.2f} queries/second")
            report.append("")
        
        report.append("=" * 65)
        return "\n".join(report)


def run_comprehensive_performance_test():
    """Run comprehensive performance test using dp_system.py queries."""
    print("Comprehensive Performance Assessment")
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
    
    print(f"\nResults saved to: {filename}")
    return results, filename


if __name__ == "__main__":
    results, filename = run_comprehensive_performance_test()
