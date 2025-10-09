"""DAX measure writer for Power BI."""

import sys
from pathlib import Path
from typing import Dict, Any, List
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class DAXWriter:
    """
    Generate DAX measures for Power BI from KPI specifications.
    """
    
    def __init__(self, kpi_specs_path: str = "src/metrics/kpis.yaml"):
        """
        Initialize DAX writer.
        
        Args:
            kpi_specs_path: Path to KPI specifications
        """
        with open(kpi_specs_path, 'r') as f:
            self.specs = yaml.safe_load(f)
    
    def generate_measure(self, kpi_name: str) -> str:
        """
        Generate DAX measure for a KPI.
        
        Args:
            kpi_name: Name of the KPI
            
        Returns:
            DAX measure string
        """
        kpi_spec = self.specs.get('kpis', {}).get(kpi_name.lower(), {})
        
        if not kpi_spec:
            return f"-- KPI '{kpi_name}' not found in specifications"
        
        formula_sql = kpi_spec.get('formula_sql', '')
        
        # Convert SQL to DAX (simplified mapping)
        dax = self._sql_to_dax(formula_sql, kpi_spec)
        
        measure_name = kpi_spec.get('name', kpi_name)
        
        return f"{measure_name} = {dax}"
    
    def _sql_to_dax(self, sql_formula: str, kpi_spec: Dict[str, Any]) -> str:
        """
        Convert SQL formula to DAX.
        
        Args:
            sql_formula: SQL formula
            kpi_spec: KPI specification
            
        Returns:
            DAX formula
        """
        # Simplified SQL to DAX conversion
        dax = sql_formula
        
        # Replace SQL aggregates with DAX
        dax = dax.replace('SUM(', 'SUMX(fact_orders, ')
        dax = dax.replace('COUNT(DISTINCT ', 'DISTINCTCOUNT(')
        dax = dax.replace('AVG(', 'AVERAGE(')
        dax = dax.replace('NULLIF(', 'IF(')
        
        # Handle column references
        dax = dax.replace('quantity * price', 'fact_orders[quantity] * fact_orders[price]')
        dax = dax.replace('quantity', 'fact_orders[quantity]')
        dax = dax.replace('price', 'fact_orders[price]')
        dax = dax.replace('cost', 'products[cost]')
        
        return dax
    
    def export_all_measures(self, output_path: str = "dashboards/powerbi/measures.txt"):
        """
        Export all KPI measures to a file.
        
        Args:
            output_path: Output file path
        """
        measures = []
        
        for kpi_name in self.specs.get('kpis', {}).keys():
            measure = self.generate_measure(kpi_name)
            measures.append(measure)
        
        # Also include predefined measures
        for measure_spec in self.specs.get('measures', []):
            measures.append(f"{measure_spec['name']} = {measure_spec['dax']}")
        
        output = "\n\n".join(measures)
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(output)
        
        return output_path


if __name__ == "__main__":
    writer = DAXWriter()
    output_path = writer.export_all_measures()
    print(f"Exported DAX measures to {output_path}")

