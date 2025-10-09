"""DSPy graders for auto-evaluation of SQL and KPI outputs."""

import dspy
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import sqlglot


@dataclass
class GradeResult:
    """Result of grading."""
    passed: bool
    score: float
    feedback: str
    details: Optional[Dict[str, Any]] = None


class GradeSQL(dspy.Signature):
    """Grade SQL query for correctness and quality."""
    
    task = dspy.InputField(desc="Original natural language task")
    generated_sql = dspy.InputField(desc="Generated SQL query")
    gold_sql = dspy.InputField(desc="Reference/gold SQL query (if available)")
    execution_result = dspy.InputField(desc="Execution result or error message")
    
    score = dspy.OutputField(desc="Quality score from 0-10")
    feedback = dspy.OutputField(desc="Detailed feedback on correctness and quality")
    issues = dspy.OutputField(desc="List of specific issues or anti-patterns")


class GradeKPI(dspy.Signature):
    """Grade KPI measure implementation."""
    
    kpi_name = dspy.InputField(desc="KPI name")
    formula_nl = dspy.InputField(desc="Natural language formula")
    generated_measure = dspy.InputField(desc="Generated SQL/DAX measure")
    test_results = dspy.InputField(desc="Validation test results")
    
    score = dspy.OutputField(desc="Quality score from 0-10")
    feedback = dspy.OutputField(desc="Feedback on measure correctness")
    improvements = dspy.OutputField(desc="Suggested improvements")


class SQLGrader(dspy.Module):
    """
    DSPy module for grading SQL query quality.
    Checks syntax, semantics, anti-patterns, and execution correctness.
    """
    
    def __init__(self):
        """Initialize SQL grader."""
        super().__init__()
        self.grade = dspy.ChainOfThought(GradeSQL)
    
    def _check_syntax(self, sql: str) -> tuple[bool, str]:
        """Check SQL syntax using sqlglot."""
        try:
            sqlglot.parse_one(sql, read='duckdb')
            return True, "Syntax valid"
        except Exception as e:
            return False, f"Syntax error: {str(e)}"
    
    def _check_anti_patterns(self, sql: str) -> List[str]:
        """Check for SQL anti-patterns."""
        issues = []
        sql_upper = sql.upper()
        
        # SELECT *
        if 'SELECT *' in sql_upper:
            issues.append("Avoid SELECT * - specify columns explicitly")
        
        # CROSS JOIN without filter
        if 'CROSS JOIN' in sql_upper and 'WHERE' not in sql_upper:
            issues.append("CROSS JOIN without WHERE clause may cause cartesian product")
        
        # Missing GROUP BY with aggregates
        if any(agg in sql_upper for agg in ['SUM(', 'AVG(', 'COUNT(', 'MAX(', 'MIN(']):
            if 'GROUP BY' not in sql_upper and 'OVER(' not in sql_upper:
                # Check if it's a scalar aggregate
                if 'SELECT' in sql_upper and sql_upper.count('SELECT') == 1:
                    pass  # Single aggregate query is OK
        
        # Non-equi joins
        if 'JOIN' in sql_upper and 'ON' in sql_upper:
            if '>=' in sql or '<=' in sql or '>' in sql or '<' in sql:
                if 'ON' in sql_upper:
                    issues.append("Non-equi join detected - verify performance")
        
        return issues
    
    def forward(self, task: str, generated_sql: str,
                gold_sql: Optional[str] = None,
                execution_result: Optional[str] = None) -> GradeResult:
        """
        Grade a generated SQL query.
        
        Args:
            task: Original NL task
            generated_sql: Generated SQL
            gold_sql: Optional reference SQL
            execution_result: Optional execution result or error
            
        Returns:
            GradeResult with score and feedback
        """
        # Static checks
        syntax_ok, syntax_msg = self._check_syntax(generated_sql)
        anti_patterns = self._check_anti_patterns(generated_sql)
        
        # Prepare execution result
        exec_result = execution_result or "No execution result provided"
        
        # Grade with LLM
        gold_sql = gold_sql or "No reference query provided"
        
        result = self.grade(
            task=task,
            generated_sql=generated_sql,
            gold_sql=gold_sql,
            execution_result=exec_result
        )
        
        # Parse score
        try:
            score = float(result.score) / 10.0  # Normalize to 0-1
        except:
            score = 0.5
        
        # Adjust score based on static checks
        if not syntax_ok:
            score *= 0.3  # Major penalty for syntax errors
        
        if anti_patterns:
            score *= 0.9  # Minor penalty per anti-pattern
        
        # Combine feedback
        feedback = result.feedback
        if anti_patterns:
            feedback += "\n\nAnti-patterns detected:\n" + "\n".join(f"- {p}" for p in anti_patterns)
        
        passed = score >= 0.7 and syntax_ok
        
        return GradeResult(
            passed=passed,
            score=score,
            feedback=feedback,
            details={
                'syntax_ok': syntax_ok,
                'syntax_msg': syntax_msg,
                'anti_patterns': anti_patterns,
                'issues': result.issues
            }
        )


class KPIGrader(dspy.Module):
    """
    DSPy module for grading KPI measure implementations.
    Checks correctness, completeness, and test coverage.
    """
    
    def __init__(self):
        """Initialize KPI grader."""
        super().__init__()
        self.grade = dspy.ChainOfThought(GradeKPI)
    
    def forward(self, kpi_name: str, formula_nl: str,
                generated_measure: str, 
                test_results: Optional[Dict[str, Any]] = None) -> GradeResult:
        """
        Grade a KPI measure implementation.
        
        Args:
            kpi_name: KPI name
            formula_nl: Natural language formula
            generated_measure: Generated SQL/DAX
            test_results: Optional test results
            
        Returns:
            GradeResult
        """
        test_results = test_results or {}
        test_str = str(test_results)
        
        result = self.grade(
            kpi_name=kpi_name,
            formula_nl=formula_nl,
            generated_measure=generated_measure,
            test_results=test_str
        )
        
        # Parse score
        try:
            score = float(result.score) / 10.0
        except:
            score = 0.5
        
        # Check test pass rate
        if test_results:
            test_pass_rate = test_results.get('pass_rate', 0)
            score = score * 0.7 + test_pass_rate * 0.3
        
        passed = score >= 0.7
        
        return GradeResult(
            passed=passed,
            score=score,
            feedback=result.feedback,
            details={
                'improvements': result.improvements,
                'test_results': test_results
            }
        )

