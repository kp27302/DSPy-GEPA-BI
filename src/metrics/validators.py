"""Data quality validation - Great Expectations lite."""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.duck import DuckDBManager


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    name: str
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None


class DataValidator:
    """
    Data quality validator for BI tables.
    Implements Great Expectations-style assertions.
    """
    
    def __init__(self, db: DuckDBManager):
        """
        Initialize validator.
        
        Args:
            db: DuckDB manager instance
        """
        self.db = db
        self.results: List[ValidationResult] = []
    
    def expect_table_exists(self, table_name: str) -> ValidationResult:
        """Check if table exists."""
        exists = self.db.table_exists(table_name)
        result = ValidationResult(
            name=f"table_exists_{table_name}",
            passed=exists,
            message=f"Table '{table_name}' exists" if exists else f"Table '{table_name}' not found"
        )
        self.results.append(result)
        return result
    
    def expect_column_values_not_null(self, table_name: str, column: str, 
                                     threshold: float = 1.0) -> ValidationResult:
        """Check that column has no (or few) null values."""
        query = f"""
        SELECT 
            COUNT(*) as total_rows,
            COUNT({column}) as non_null_rows,
            COUNT(*) - COUNT({column}) as null_rows
        FROM {table_name}
        """
        result_df = self.db.query_df(query)
        
        total = int(result_df['total_rows'].iloc[0])
        non_null = int(result_df['non_null_rows'].iloc[0])
        null_count = int(result_df['null_rows'].iloc[0])
        
        non_null_pct = non_null / total if total > 0 else 0
        passed = non_null_pct >= threshold
        
        result = ValidationResult(
            name=f"not_null_{table_name}.{column}",
            passed=passed,
            message=f"{column}: {non_null_pct*100:.1f}% non-null (threshold: {threshold*100:.1f}%)",
            details={'null_count': null_count, 'total': total, 'non_null_pct': non_null_pct}
        )
        self.results.append(result)
        return result
    
    def expect_column_values_to_be_unique(self, table_name: str, column: str) -> ValidationResult:
        """Check that column values are unique."""
        query = f"""
        SELECT 
            COUNT(*) as total_rows,
            COUNT(DISTINCT {column}) as unique_values
        FROM {table_name}
        """
        result_df = self.db.query_df(query)
        
        total = int(result_df['total_rows'].iloc[0])
        unique = int(result_df['unique_values'].iloc[0])
        
        passed = total == unique
        
        result = ValidationResult(
            name=f"unique_{table_name}.{column}",
            passed=passed,
            message=f"{column}: {unique} unique of {total} total",
            details={'total': total, 'unique': unique, 'duplicates': total - unique}
        )
        self.results.append(result)
        return result
    
    def expect_column_values_in_range(self, table_name: str, column: str,
                                     min_value: Optional[float] = None,
                                     max_value: Optional[float] = None) -> ValidationResult:
        """Check that numeric column values are within range."""
        conditions = []
        if min_value is not None:
            conditions.append(f"{column} >= {min_value}")
        if max_value is not None:
            conditions.append(f"{column} <= {max_value}")
        
        where_clause = " AND ".join(conditions)
        
        query = f"""
        SELECT 
            COUNT(*) as total_rows,
            SUM(CASE WHEN {where_clause} THEN 1 ELSE 0 END) as in_range_rows,
            MIN({column}) as min_val,
            MAX({column}) as max_val
        FROM {table_name}
        WHERE {column} IS NOT NULL
        """
        result_df = self.db.query_df(query)
        
        total = int(result_df['total_rows'].iloc[0])
        in_range = int(result_df['in_range_rows'].iloc[0])
        actual_min = float(result_df['min_val'].iloc[0])
        actual_max = float(result_df['max_val'].iloc[0])
        
        passed = total == in_range
        
        result = ValidationResult(
            name=f"range_{table_name}.{column}",
            passed=passed,
            message=f"{column}: {in_range}/{total} in range [{min_value}, {max_value}]",
            details={
                'in_range': in_range,
                'total': total,
                'actual_min': actual_min,
                'actual_max': actual_max
            }
        )
        self.results.append(result)
        return result
    
    def expect_foreign_key_integrity(self, table_name: str, column: str,
                                    ref_table: str, ref_column: str) -> ValidationResult:
        """Check foreign key referential integrity."""
        query = f"""
        SELECT COUNT(*) as orphaned_rows
        FROM {table_name} t
        WHERE t.{column} IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM {ref_table} r
              WHERE r.{ref_column} = t.{column}
          )
        """
        result_df = self.db.query_df(query)
        orphaned = int(result_df['orphaned_rows'].iloc[0])
        
        passed = orphaned == 0
        
        result = ValidationResult(
            name=f"fk_{table_name}.{column}",
            passed=passed,
            message=f"{column} → {ref_table}.{ref_column}: {orphaned} orphaned rows",
            details={'orphaned_count': orphaned}
        )
        self.results.append(result)
        return result
    
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        
        return {
            'total_tests': total,
            'passed': passed,
            'failed': total - passed,
            'pass_rate': passed / total if total > 0 else 0,
            'results': [
                {
                    'name': r.name,
                    'passed': r.passed,
                    'message': r.message,
                    'details': r.details
                }
                for r in self.results
            ]
        }
    
    def clear_results(self):
        """Clear validation results."""
        self.results = []

