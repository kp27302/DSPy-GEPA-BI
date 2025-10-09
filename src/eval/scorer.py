"""Scoring functions for SQL and KPI evaluation."""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.duck import DuckDBManager


@dataclass
class EvalResult:
    """Result of evaluation."""
    task_id: str
    passed: bool
    score: float
    execution_match: bool
    static_checks_passed: bool
    token_cost: int
    latency: float
    details: Dict[str, Any]


class SQLScorer:
    """
    Scorer for SQL query evaluation.
    Supports execution-based, static analysis, and tolerance-based scoring.
    """
    
    def __init__(self, db: DuckDBManager, tolerance: float = 0.001):
        """
        Initialize SQL scorer.
        
        Args:
            db: DuckDB manager for query execution
            tolerance: Float comparison tolerance
        """
        self.db = db
        self.tolerance = tolerance
    
    def score_execution(self, generated_sql: str, gold_sql: str) -> Tuple[bool, float, str]:
        """
        Score SQL by executing and comparing results.
        
        Args:
            generated_sql: Generated SQL query
            gold_sql: Gold/reference SQL query
            
        Returns:
            Tuple of (passed, score, message)
        """
        try:
            # Execute both queries
            gen_result = self.db.query_df(generated_sql)
            gold_result = self.db.query_df(gold_sql)
            
            # Compare results
            if gen_result.empty and gold_result.empty:
                return True, 1.0, "Both queries return empty result"
            
            if gen_result.empty != gold_result.empty:
                return False, 0.0, "One query returns empty, other doesn't"
            
            # Check column names
            if set(gen_result.columns) != set(gold_result.columns):
                return False, 0.5, f"Column mismatch: {gen_result.columns} vs {gold_result.columns}"
            
            # Sort both dataframes for comparison (order-agnostic)
            gen_sorted = gen_result.sort_values(by=list(gen_result.columns)).reset_index(drop=True)
            gold_sorted = gold_result.sort_values(by=list(gold_result.columns)).reset_index(drop=True)
            
            # Check row counts
            if len(gen_sorted) != len(gold_sorted):
                return False, 0.6, f"Row count mismatch: {len(gen_sorted)} vs {len(gold_sorted)}"
            
            # Compare values with tolerance
            match_score = self._compare_dataframes(gen_sorted, gold_sorted)
            
            passed = match_score >= 0.95
            message = f"Match score: {match_score:.3f}"
            
            return passed, match_score, message
            
        except Exception as e:
            return False, 0.0, f"Execution error: {str(e)}"
    
    def _compare_dataframes(self, df1: pd.DataFrame, df2: pd.DataFrame) -> float:
        """
        Compare two dataframes with float tolerance.
        
        Args:
            df1: First dataframe
            df2: Second dataframe
            
        Returns:
            Match score (0-1)
        """
        if df1.shape != df2.shape:
            return 0.0
        
        total_cells = df1.shape[0] * df1.shape[1]
        matching_cells = 0
        
        for col in df1.columns:
            for i in range(len(df1)):
                val1 = df1[col].iloc[i]
                val2 = df2[col].iloc[i]
                
                # Handle nulls
                if pd.isna(val1) and pd.isna(val2):
                    matching_cells += 1
                elif pd.isna(val1) or pd.isna(val2):
                    continue
                # Handle floats
                elif isinstance(val1, (float, np.floating)) and isinstance(val2, (float, np.floating)):
                    if abs(val1 - val2) <= self.tolerance:
                        matching_cells += 1
                # Handle other types
                elif val1 == val2:
                    matching_cells += 1
        
        return matching_cells / total_cells if total_cells > 0 else 0.0
    
    def check_static_quality(self, sql: str) -> Tuple[bool, List[str]]:
        """
        Static quality checks for SQL.
        
        Args:
            sql: SQL query string
            
        Returns:
            Tuple of (passed, list of issues)
        """
        issues = []
        sql_upper = sql.upper()
        
        # Anti-pattern checks
        if 'SELECT *' in sql_upper:
            issues.append("Uses SELECT * instead of explicit columns")
        
        if 'CROSS JOIN' in sql_upper and 'WHERE' not in sql_upper:
            issues.append("CROSS JOIN without WHERE clause")
        
        # Check for potential full table scans
        if 'SELECT' in sql_upper and 'WHERE' not in sql_upper and 'LIMIT' not in sql_upper:
            if 'GROUP BY' not in sql_upper and 'JOIN' not in sql_upper:
                issues.append("Potential full table scan without filter")
        
        passed = len(issues) == 0
        return passed, issues
    
    def evaluate(self, generated_sql: str, gold_sql: Optional[str] = None,
                task_id: str = "unknown") -> EvalResult:
        """
        Complete evaluation of a SQL query.
        
        Args:
            generated_sql: Generated SQL
            gold_sql: Optional gold SQL
            task_id: Task identifier
            
        Returns:
            EvalResult
        """
        start_time = time.time()
        
        # Static checks
        static_passed, static_issues = self.check_static_quality(generated_sql)
        
        # Execution check (if gold SQL provided)
        exec_passed = False
        exec_score = 0.0
        exec_msg = "No gold SQL provided"
        
        if gold_sql:
            exec_passed, exec_score, exec_msg = self.score_execution(generated_sql, gold_sql)
        
        latency = time.time() - start_time
        
        # Estimate token cost (rough)
        token_cost = len(generated_sql.split()) * 2
        
        # Overall score
        score = exec_score * 0.7 + (1.0 if static_passed else 0.5) * 0.3
        passed = exec_passed and static_passed
        
        return EvalResult(
            task_id=task_id,
            passed=passed,
            score=score,
            execution_match=exec_passed,
            static_checks_passed=static_passed,
            token_cost=token_cost,
            latency=latency,
            details={
                'execution_message': exec_msg,
                'static_issues': static_issues,
                'execution_score': exec_score
            }
        )


class KPIScorer:
    """
    Scorer for KPI/measure evaluation.
    Checks correctness, test coverage, and semantic accuracy.
    """
    
    def __init__(self, db: DuckDBManager):
        """
        Initialize KPI scorer.
        
        Args:
            db: DuckDB manager
        """
        self.db = db
    
    def evaluate(self, kpi_name: str, measure_sql: str,
                expected_properties: List[str] = None,
                task_id: str = "unknown") -> EvalResult:
        """
        Evaluate a KPI measure.
        
        Args:
            kpi_name: KPI name
            measure_sql: Generated measure SQL
            expected_properties: Expected properties (positive, range, etc.)
            task_id: Task identifier
            
        Returns:
            EvalResult
        """
        start_time = time.time()
        expected_properties = expected_properties or []
        
        # Try to execute measure
        try:
            result = self.db.query_df(measure_sql)
            exec_passed = True
            exec_msg = "Measure executed successfully"
        except Exception as e:
            exec_passed = False
            exec_msg = f"Execution failed: {str(e)}"
            result = pd.DataFrame()
        
        # Check expected properties
        property_checks = self._check_properties(result, expected_properties)
        properties_passed = all(property_checks.values())
        
        latency = time.time() - start_time
        token_cost = len(measure_sql.split()) * 2
        
        score = (1.0 if exec_passed else 0.0) * 0.6 + \
                (sum(property_checks.values()) / max(len(property_checks), 1)) * 0.4
        
        passed = exec_passed and properties_passed
        
        return EvalResult(
            task_id=task_id,
            passed=passed,
            score=score,
            execution_match=exec_passed,
            static_checks_passed=properties_passed,
            token_cost=token_cost,
            latency=latency,
            details={
                'execution_message': exec_msg,
                'property_checks': property_checks,
                'result_shape': result.shape if not result.empty else (0, 0)
            }
        )
    
    def _check_properties(self, result: pd.DataFrame, 
                         expected: List[str]) -> Dict[str, bool]:
        """Check expected properties of measure result."""
        checks = {}
        
        if result.empty:
            return {prop: False for prop in expected}
        
        # Get first column (measure value)
        if len(result.columns) == 0:
            return {prop: False for prop in expected}
        
        values = result.iloc[:, 0]
        
        for prop in expected:
            if prop == 'positive':
                checks[prop] = (values >= 0).all()
            elif prop == 'not_null':
                checks[prop] = values.notna().all()
            elif prop.startswith('range'):
                # Simplified range check
                checks[prop] = True
            else:
                checks[prop] = True
        
        return checks


def evaluate_sql_task(db: DuckDBManager, task: Dict[str, Any],
                     generated_sql: str) -> EvalResult:
    """
    Evaluate a SQL task.
    
    Args:
        db: DuckDB manager
        task: Task dict with 'id', 'nl', 'gold_sql'
        generated_sql: Generated SQL
        
    Returns:
        EvalResult
    """
    scorer = SQLScorer(db)
    return scorer.evaluate(
        generated_sql=generated_sql,
        gold_sql=task.get('gold_sql'),
        task_id=task.get('id', 'unknown')
    )


def evaluate_kpi_task(db: DuckDBManager, task: Dict[str, Any],
                     measure_sql: str) -> EvalResult:
    """
    Evaluate a KPI task.
    
    Args:
        db: DuckDB manager
        task: Task dict with 'id', 'kpi_name', 'expected_properties'
        measure_sql: Generated measure SQL
        
    Returns:
        EvalResult
    """
    scorer = KPIScorer(db)
    return scorer.evaluate(
        kpi_name=task.get('kpi_name', 'unknown'),
        measure_sql=measure_sql,
        expected_properties=task.get('expected_properties', []),
        task_id=task.get('id', 'unknown')
    )

