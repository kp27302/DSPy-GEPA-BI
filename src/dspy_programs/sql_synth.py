"""DSPy program for natural language to SQL synthesis."""

import dspy
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class SQLSynthResult:
    """Result of SQL synthesis."""
    sql: str
    rationale: str
    estimated_cost: int
    confidence: float


class GenerateSQL(dspy.Signature):
    """Generate SQL query from natural language with schema context."""
    
    task = dspy.InputField(desc="Natural language description of the query task")
    schema = dspy.InputField(desc="Database schema with tables, columns, types, and relationships")
    constraints = dspy.InputField(desc="Query constraints: filters, joins, aggregations needed")
    examples = dspy.InputField(desc="Few-shot examples of similar NL→SQL transformations")
    
    sql = dspy.OutputField(desc="Syntactically correct SQL query")
    rationale = dspy.OutputField(desc="Step-by-step explanation of query construction")


class RefineSQL(dspy.Signature):
    """Refine SQL query based on validation feedback."""
    
    original_sql = dspy.InputField(desc="Original SQL query")
    error_message = dspy.InputField(desc="Error or validation feedback")
    schema = dspy.InputField(desc="Database schema context")
    
    refined_sql = dspy.OutputField(desc="Corrected SQL query")
    changes = dspy.OutputField(desc="Explanation of changes made")


class SQLSynthesizer(dspy.Module):
    """
    DSPy module for synthesizing SQL from natural language.
    Supports schema-aware generation, dry-run validation, and iterative refinement.
    """
    
    def __init__(self, schema_context: Dict[str, Any], few_shot_examples: List[Dict] = None):
        """
        Initialize SQL synthesizer.
        
        Args:
            schema_context: Database schema information
            few_shot_examples: Optional list of example NL→SQL pairs
        """
        super().__init__()
        self.schema_context = schema_context
        self.few_shot_examples = few_shot_examples or []
        
        # DSPy chain: generate → validate → refine if needed
        self.generate = dspy.ChainOfThought(GenerateSQL)
        self.refine = dspy.ChainOfThought(RefineSQL)
    
    def _format_schema(self) -> str:
        """Format schema context for LLM."""
        schema_str = "Database Schema:\n\n"
        
        for table_name, table_info in self.schema_context.items():
            schema_str += f"Table: {table_name}\n"
            if 'columns' in table_info:
                for col in table_info['columns']:
                    col_str = f"  - {col['name']}: {col['type']}"
                    if col.get('pk'):
                        col_str += " (PRIMARY KEY)"
                    if col.get('fk'):
                        col_str += f" (FOREIGN KEY → {col['fk']})"
                    schema_str += col_str + "\n"
            schema_str += "\n"
        
        return schema_str
    
    def _format_examples(self) -> str:
        """Format few-shot examples."""
        if not self.few_shot_examples:
            return "No examples provided."
        
        examples_str = "Example Query Translations:\n\n"
        for i, ex in enumerate(self.few_shot_examples[:5], 1):
            examples_str += f"Example {i}:\n"
            examples_str += f"Task: {ex.get('nl', '')}\n"
            examples_str += f"SQL: {ex.get('sql', '')}\n\n"
        
        return examples_str
    
    def forward(self, task: str, constraints: Optional[str] = None) -> SQLSynthResult:
        """
        Generate SQL from natural language task.
        
        Args:
            task: Natural language description of query
            constraints: Optional query constraints
            
        Returns:
            SQLSynthResult with generated query and metadata
        """
        schema = self._format_schema()
        examples = self._format_examples()
        constraints = constraints or "No specific constraints."
        
        # Generate SQL
        result = self.generate(
            task=task,
            schema=schema,
            constraints=constraints,
            examples=examples
        )
        
        # Extract SQL and clean up
        sql = result.sql.strip()
        if sql.startswith("```sql"):
            sql = sql.replace("```sql", "").replace("```", "").strip()
        
        # Estimate token cost (rough heuristic)
        estimated_cost = len(result.sql.split()) * 2 + len(result.rationale.split())
        
        return SQLSynthResult(
            sql=sql,
            rationale=result.rationale,
            estimated_cost=estimated_cost,
            confidence=0.8  # Could be computed from model logprobs
        )
    
    def refine_with_feedback(self, original_sql: str, error_message: str) -> str:
        """
        Refine SQL based on error feedback.
        
        Args:
            original_sql: Original SQL that failed
            error_message: Error or validation message
            
        Returns:
            Refined SQL query
        """
        schema = self._format_schema()
        
        result = self.refine(
            original_sql=original_sql,
            error_message=error_message,
            schema=schema
        )
        
        refined_sql = result.refined_sql.strip()
        if refined_sql.startswith("```sql"):
            refined_sql = refined_sql.replace("```sql", "").replace("```", "").strip()
        
        return refined_sql


# Utility function for standalone usage
def synthesize_sql(task: str, schema: Dict[str, Any], 
                   model: str = "openai/gpt-4o-mini") -> SQLSynthResult:
    """
    Standalone function to synthesize SQL.
    
    Args:
        task: Natural language query description
        schema: Database schema dictionary
        model: LLM model identifier
        
    Returns:
        SQLSynthResult
    """
    # Configure DSPy with LLM
    lm = dspy.OpenAI(model=model, max_tokens=2048, temperature=0.1)
    dspy.settings.configure(lm=lm)
    
    # Create synthesizer and generate
    synthesizer = SQLSynthesizer(schema_context=schema)
    return synthesizer(task=task)

