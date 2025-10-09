"""Reflective learning: mine rules from execution traces."""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import Counter
import re


@dataclass
class Rule:
    """A mined rule from execution traces."""
    rule_id: str
    pattern: str
    action: str
    confidence: float
    support: int
    examples: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'rule_id': self.rule_id,
            'pattern': self.pattern,
            'action': self.action,
            'confidence': self.confidence,
            'support': self.support,
            'examples': self.examples or []
        }


class Reflector:
    """
    Analyzes execution traces to extract learnings.
    Identifies patterns in failures and successes.
    """
    
    def __init__(self):
        """Initialize reflector."""
        self.traces: List[Dict[str, Any]] = []
        self.rules: List[Rule] = []
    
    def add_trace(self, trace: Dict[str, Any]):
        """
        Add execution trace.
        
        Args:
            trace: Dictionary with task, generated output, result, errors, etc.
        """
        self.traces.append(trace)
    
    def reflect(self) -> List[Rule]:
        """
        Analyze traces and extract rules.
        
        Returns:
            List of mined rules
        """
        self.rules = []
        
        # Analyze failures
        failures = [t for t in self.traces if not t.get('passed', False)]
        successes = [t for t in self.traces if t.get('passed', False)]
        
        # Mine failure patterns
        if failures:
            self.rules.extend(self._mine_failure_patterns(failures))
        
        # Mine success patterns
        if successes:
            self.rules.extend(self._mine_success_patterns(successes))
        
        return self.rules
    
    def _mine_failure_patterns(self, failures: List[Dict[str, Any]]) -> List[Rule]:
        """Mine patterns from failures."""
        rules = []
        
        # Common error patterns
        error_patterns = Counter()
        
        for trace in failures:
            error = trace.get('error', '')
            feedback = trace.get('feedback', '')
            
            # Pattern: CROSS JOIN without filter
            if 'CROSS JOIN' in trace.get('generated_sql', '').upper():
                if 'WHERE' not in trace.get('generated_sql', '').upper():
                    error_patterns['cross_join_no_filter'] += 1
            
            # Pattern: Missing GROUP BY
            if 'GROUP BY' in error.lower() or 'must appear in group by' in error.lower():
                error_patterns['missing_group_by'] += 1
            
            # Pattern: Ambiguous column
            if 'ambiguous' in error.lower():
                error_patterns['ambiguous_column'] += 1
            
            # Pattern: Unknown table/column
            if 'does not exist' in error.lower() or 'unknown' in error.lower():
                error_patterns['unknown_identifier'] += 1
        
        # Create rules from patterns
        for pattern, count in error_patterns.items():
            if count >= 2:  # Minimum support
                rule = self._pattern_to_rule(pattern, count, len(failures))
                if rule:
                    rules.append(rule)
        
        return rules
    
    def _mine_success_patterns(self, successes: List[Dict[str, Any]]) -> List[Rule]:
        """Mine patterns from successful executions."""
        rules = []
        
        # Look for common good practices
        practice_patterns = Counter()
        
        for trace in successes:
            sql = trace.get('generated_sql', '').upper()
            
            # Pattern: Explicit column selection
            if 'SELECT *' not in sql:
                practice_patterns['explicit_columns'] += 1
            
            # Pattern: Using INNER JOIN
            if 'INNER JOIN' in sql:
                practice_patterns['prefer_inner_join'] += 1
            
            # Pattern: Proper GROUP BY
            if 'GROUP BY' in sql and 'SUM(' in sql or 'AVG(' in sql:
                practice_patterns['proper_aggregation'] += 1
        
        # Create positive rules
        for pattern, count in practice_patterns.items():
            if count / len(successes) >= 0.7:  # 70% of successes
                rule = self._pattern_to_positive_rule(pattern, count, len(successes))
                if rule:
                    rules.append(rule)
        
        return rules
    
    def _pattern_to_rule(self, pattern: str, support: int, total: int) -> Optional[Rule]:
        """Convert error pattern to rule."""
        confidence = support / total if total > 0 else 0
        
        rule_map = {
            'cross_join_no_filter': Rule(
                rule_id='avoid_unfiltered_cross_join',
                pattern='CROSS JOIN without WHERE clause',
                action='Add WHERE clause or use INNER/LEFT JOIN instead',
                confidence=confidence,
                support=support
            ),
            'missing_group_by': Rule(
                rule_id='add_group_by',
                pattern='Aggregate function without GROUP BY',
                action='Add GROUP BY for non-aggregated columns in SELECT',
                confidence=confidence,
                support=support
            ),
            'ambiguous_column': Rule(
                rule_id='qualify_columns',
                pattern='Ambiguous column reference',
                action='Qualify column names with table aliases',
                confidence=confidence,
                support=support
            ),
            'unknown_identifier': Rule(
                rule_id='check_schema',
                pattern='Unknown table or column',
                action='Verify identifiers against schema before using',
                confidence=confidence,
                support=support
            )
        }
        
        return rule_map.get(pattern)
    
    def _pattern_to_positive_rule(self, pattern: str, support: int, total: int) -> Optional[Rule]:
        """Convert success pattern to positive rule."""
        confidence = support / total if total > 0 else 0
        
        rule_map = {
            'explicit_columns': Rule(
                rule_id='use_explicit_columns',
                pattern='SELECT with explicit column list',
                action='Continue using explicit column selection',
                confidence=confidence,
                support=support
            ),
            'prefer_inner_join': Rule(
                rule_id='prefer_inner_join',
                pattern='Using INNER JOIN',
                action='Prefer INNER JOIN for better performance',
                confidence=confidence,
                support=support
            ),
            'proper_aggregation': Rule(
                rule_id='proper_group_by',
                pattern='Correct GROUP BY with aggregates',
                action='Continue proper GROUP BY usage',
                confidence=confidence,
                support=support
            )
        }
        
        return rule_map.get(pattern)


class RuleMiner:
    """
    Advanced rule mining from execution traces.
    Supports association rule mining and pattern extraction.
    """
    
    def __init__(self, min_confidence: float = 0.5, min_support: int = 2):
        """
        Initialize rule miner.
        
        Args:
            min_confidence: Minimum confidence threshold
            min_support: Minimum support count
        """
        self.min_confidence = min_confidence
        self.min_support = min_support
    
    def mine_rules(self, traces: List[Dict[str, Any]]) -> List[Rule]:
        """
        Mine association rules from traces.
        
        Args:
            traces: List of execution traces
            
        Returns:
            List of mined rules
        """
        reflector = Reflector()
        for trace in traces:
            reflector.add_trace(trace)
        
        rules = reflector.reflect()
        
        # Filter by confidence and support
        filtered_rules = [
            r for r in rules
            if r.confidence >= self.min_confidence and r.support >= self.min_support
        ]
        
        return filtered_rules

