"""DSPy program for compiling KPI definitions into SQL/DAX measures."""

import dspy
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class KPICompileResult:
    """Result of KPI compilation."""
    measure_sql: str
    measure_dax: str
    tests: List[Dict[str, Any]]
    grain: List[str]
    dependencies: List[str]


class CompileKPI(dspy.Signature):
    """Compile KPI definition into executable measure."""
    
    kpi_name = dspy.InputField(desc="Name of the KPI metric")
    formula_nl = dspy.InputField(desc="Natural language formula description")
    grain = dspy.InputField(desc="Aggregation grain: date, region, segment, etc.")
    schema = dspy.InputField(desc="Database schema with fact and dimension tables")
    
    measure_sql = dspy.OutputField(desc="SQL implementation of the KPI measure")
    measure_dax = dspy.OutputField(desc="DAX implementation for Power BI")
    rationale = dspy.OutputField(desc="Explanation of the measure logic")


class GenerateKPITests(dspy.Signature):
    """Generate data quality tests for a KPI."""
    
    kpi_name = dspy.InputField(desc="KPI name")
    measure_sql = dspy.InputField(desc="SQL measure definition")
    expected_properties = dspy.InputField(desc="Expected properties: positive, range, etc.")
    
    tests = dspy.OutputField(desc="List of validation tests in YAML format")


class KPICompiler(dspy.Module):
    """
    DSPy module for compiling KPI definitions into executable measures.
    Generates both SQL and DAX, with accompanying validation tests.
    """
    
    def __init__(self, schema_context: Dict[str, Any], kpi_specs: Dict[str, Any]):
        """
        Initialize KPI compiler.
        
        Args:
            schema_context: Database schema
            kpi_specs: KPI specifications from kpis.yaml
        """
        super().__init__()
        self.schema_context = schema_context
        self.kpi_specs = kpi_specs
        
        self.compile_kpi = dspy.ChainOfThought(CompileKPI)
        self.generate_tests = dspy.ChainOfThought(GenerateKPITests)
    
    def _format_schema(self) -> str:
        """Format schema for context."""
        schema_str = "Available Tables:\n\n"
        
        for table_name, table_info in self.schema_context.items():
            schema_str += f"- {table_name}: "
            if 'columns' in table_info:
                cols = [col['name'] for col in table_info['columns']]
                schema_str += ", ".join(cols)
            schema_str += "\n"
        
        return schema_str
    
    def forward(self, kpi_name: str) -> KPICompileResult:
        """
        Compile a KPI into SQL and DAX measures with tests.
        
        Args:
            kpi_name: Name of the KPI to compile
            
        Returns:
            KPICompileResult with SQL, DAX, and tests
        """
        # Get KPI spec
        kpi_spec = self.kpi_specs.get(kpi_name.lower(), {})
        if not kpi_spec:
            raise ValueError(f"KPI '{kpi_name}' not found in specifications")
        
        formula_nl = kpi_spec.get('formula_nl', '')
        grain = kpi_spec.get('grain', [])
        
        # Compile KPI
        result = self.compile_kpi(
            kpi_name=kpi_name,
            formula_nl=formula_nl,
            grain=str(grain),
            schema=self._format_schema()
        )
        
        # Generate tests
        expected_props = str(kpi_spec.get('tests', []))
        test_result = self.generate_tests(
            kpi_name=kpi_name,
            measure_sql=result.measure_sql,
            expected_properties=expected_props
        )
        
        # Parse tests (simplified - in production, parse YAML)
        tests = [
            {'type': 'not_null', 'column': kpi_name.lower()},
            {'type': 'positive', 'column': kpi_name.lower()} if 'positive' in expected_props else None
        ]
        tests = [t for t in tests if t is not None]
        
        return KPICompileResult(
            measure_sql=result.measure_sql.strip(),
            measure_dax=result.measure_dax.strip(),
            tests=tests,
            grain=grain if isinstance(grain, list) else [grain],
            dependencies=self._extract_dependencies(result.measure_sql)
        )
    
    def _extract_dependencies(self, sql: str) -> List[str]:
        """Extract table/column dependencies from SQL."""
        # Simplified - in production use SQL parser
        dependencies = []
        sql_upper = sql.upper()
        
        for table_name in self.schema_context.keys():
            if table_name.upper() in sql_upper:
                dependencies.append(table_name)
        
        return dependencies


def compile_kpi(kpi_name: str, kpi_specs: Dict[str, Any], 
                schema: Dict[str, Any], model: str = "openai/gpt-4o-mini") -> KPICompileResult:
    """
    Standalone function to compile a KPI.
    
    Args:
        kpi_name: KPI to compile
        kpi_specs: KPI specifications
        schema: Database schema
        model: LLM model
        
    Returns:
        KPICompileResult
    """
    lm = dspy.OpenAI(model=model, max_tokens=2048, temperature=0.1)
    dspy.settings.configure(lm=lm)
    
    compiler = KPICompiler(schema_context=schema, kpi_specs=kpi_specs)
    return compiler(kpi_name=kpi_name)

