"""I/O utilities: path management, schema registry, caching."""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
import orjson


class PathManager:
    """Centralized path management for the project."""
    
    def __init__(self, root: Optional[Path] = None):
        """
        Initialize path manager.
        
        Args:
            root: Project root directory. Defaults to current working directory.
        """
        self.root = root or Path.cwd()
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        dirs = [
            "data/raw",
            "data/warehouse",
            "eval/benchmarks",
            "eval/results",
            "logs",
            "configs",
            "notebooks",
            "dashboards/powerbi",
            "dashboards/lookml",
        ]
        for d in dirs:
            (self.root / d).mkdir(parents=True, exist_ok=True)
    
    @property
    def data_raw(self) -> Path:
        """Path to raw data directory."""
        return self.root / "data" / "raw"
    
    @property
    def data_warehouse(self) -> Path:
        """Path to warehouse directory."""
        return self.root / "data" / "warehouse"
    
    @property
    def eval_benchmarks(self) -> Path:
        """Path to evaluation benchmarks."""
        return self.root / "eval" / "benchmarks"
    
    @property
    def eval_results(self) -> Path:
        """Path to evaluation results."""
        return self.root / "eval" / "results"
    
    @property
    def configs(self) -> Path:
        """Path to config directory."""
        return self.root / "configs"
    
    @property
    def logs(self) -> Path:
        """Path to logs directory."""
        return self.root / "logs"
    
    def get_warehouse_db(self) -> Path:
        """Get path to main DuckDB warehouse."""
        return self.data_warehouse / "bi.duckdb"


class SchemaRegistry:
    """Registry for database schemas with validation and introspection."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize schema registry.
        
        Args:
            config_path: Path to project.yaml config file
        """
        self.config_path = config_path or Path("configs/project.yaml")
        self.schemas = self._load_schemas()
    
    def _load_schemas(self) -> Dict[str, Any]:
        """Load schema definitions from config."""
        if not self.config_path.exists():
            return {}
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config.get('schema', {})
    
    def get_table_schema(self, table_name: str) -> Optional[Dict[str, Any]]:
        """
        Get schema definition for a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Schema dictionary or None if not found
        """
        return self.schemas.get(table_name)
    
    def get_columns(self, table_name: str) -> List[Dict[str, str]]:
        """
        Get column definitions for a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            List of column dictionaries
        """
        schema = self.get_table_schema(table_name)
        if not schema:
            return []
        return schema.get('columns', [])
    
    def get_primary_keys(self, table_name: str) -> List[str]:
        """
        Get primary key columns for a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            List of primary key column names
        """
        columns = self.get_columns(table_name)
        return [col['name'] for col in columns if col.get('pk', False)]
    
    def get_foreign_keys(self, table_name: str) -> Dict[str, str]:
        """
        Get foreign key mappings for a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dictionary mapping local column to foreign table.column
        """
        columns = self.get_columns(table_name)
        fks = {}
        for col in columns:
            if 'fk' in col:
                fks[col['name']] = col['fk']
        return fks
    
    def to_ddl(self, table_name: str) -> str:
        """
        Generate CREATE TABLE DDL for a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            DDL string
        """
        schema = self.get_table_schema(table_name)
        if not schema:
            return ""
        
        columns = schema.get('columns', [])
        col_defs = []
        
        for col in columns:
            col_def = f"{col['name']} {col['type']}"
            if col.get('pk'):
                col_def += " PRIMARY KEY"
            col_defs.append(col_def)
        
        ddl = f"CREATE TABLE {table_name} (\n  "
        ddl += ",\n  ".join(col_defs)
        ddl += "\n);"
        
        return ddl
    
    def to_json_schema(self) -> str:
        """
        Export all schemas as JSON for LLM context.
        
        Returns:
            JSON string representation of schemas
        """
        return json.dumps(self.schemas, indent=2)


def load_config(config_path: str = "configs/project.yaml") -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_json(data: Any, path: Path, fast: bool = True) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to serialize
        path: Output file path
        fast: Use orjson for faster serialization
    """
    if fast:
        with open(path, 'wb') as f:
            f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
    else:
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


def load_json(path: Path, fast: bool = True) -> Any:
    """
    Load data from JSON file.
    
    Args:
        path: Input file path
        fast: Use orjson for faster parsing
        
    Returns:
        Deserialized data
    """
    if fast:
        with open(path, 'rb') as f:
            return orjson.loads(f.read())
    else:
        with open(path, 'r') as f:
            return json.load(f)

