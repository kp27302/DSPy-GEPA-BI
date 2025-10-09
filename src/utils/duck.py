"""DuckDB connection management and query helpers."""

import duckdb
from pathlib import Path
from typing import Any, List, Dict, Optional, Union
import pandas as pd
from contextlib import contextmanager


class DuckDBManager:
    """
    Centralized DuckDB connection and query management.
    Supports connection pooling, query execution, and schema introspection.
    """
    
    def __init__(self, db_path: Union[str, Path]):
        """
        Initialize DuckDB manager.
        
        Args:
            db_path: Path to DuckDB database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[duckdb.DuckDBPyConnection] = None
    
    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = duckdb.connect(str(self.db_path))
            self._configure_connection()
        return self._conn
    
    def _configure_connection(self):
        """Configure connection with performance settings."""
        try:
            self.conn.execute("SET enable_progress_bar=false;")
        except:
            pass  # Ignore if not supported
        try:
            self.conn.execute("SET threads=4;")
        except:
            pass  # Ignore if not supported
    
    @contextmanager
    def transaction(self):
        """Context manager for transactions."""
        try:
            self.conn.execute("BEGIN TRANSACTION;")
            yield self.conn
            self.conn.execute("COMMIT;")
        except Exception as e:
            self.conn.execute("ROLLBACK;")
            raise e
    
    def execute(self, query: str, params: Optional[tuple] = None) -> duckdb.DuckDBPyConnection:
        """
        Execute a SQL query.
        
        Args:
            query: SQL query string
            params: Optional query parameters
            
        Returns:
            DuckDB connection with results
        """
        if params:
            return self.conn.execute(query, params)
        return self.conn.execute(query)
    
    def query_df(self, query: str, params: Optional[tuple] = None) -> pd.DataFrame:
        """
        Execute query and return results as DataFrame.
        
        Args:
            query: SQL query string
            params: Optional query parameters
            
        Returns:
            Query results as pandas DataFrame
        """
        result = self.execute(query, params)
        return result.df()
    
    def query_dict(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """
        Execute query and return results as list of dictionaries.
        
        Args:
            query: SQL query string
            params: Optional query parameters
            
        Returns:
            Query results as list of dicts
        """
        df = self.query_df(query, params)
        return df.to_dict('records')
    
    def load_csv(self, csv_path: Union[str, Path], table_name: str, 
                 auto_detect: bool = True) -> None:
        """
        Load CSV file into a table.
        
        Args:
            csv_path: Path to CSV file
            table_name: Target table name
            auto_detect: Auto-detect CSV format and types
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        query = f"""
        CREATE OR REPLACE TABLE {table_name} AS 
        SELECT * FROM read_csv_auto('{csv_path}', 
                                    AUTO_DETECT={str(auto_detect).upper()})
        """
        self.execute(query)
    
    def export_parquet(self, table_name: str, output_path: Union[str, Path]) -> None:
        """
        Export table to Parquet file.
        
        Args:
            table_name: Source table name
            output_path: Output Parquet file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        query = f"COPY {table_name} TO '{output_path}' (FORMAT PARQUET)"
        self.execute(query)
    
    def load_parquet(self, parquet_path: Union[str, Path], table_name: str) -> None:
        """
        Load Parquet file into a table.
        
        Args:
            parquet_path: Path to Parquet file
            table_name: Target table name
        """
        parquet_path = Path(parquet_path)
        query = f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM '{parquet_path}'"
        self.execute(query)
    
    def get_tables(self) -> List[str]:
        """
        Get list of all tables in the database.
        
        Returns:
            List of table names
        """
        result = self.query_df("SHOW TABLES")
        return result['name'].tolist() if not result.empty else []
    
    def get_table_info(self, table_name: str) -> pd.DataFrame:
        """
        Get schema information for a table.
        
        Args:
            table_name: Table name
            
        Returns:
            DataFrame with column information
        """
        query = f"DESCRIBE {table_name}"
        return self.query_df(query)
    
    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists.
        
        Args:
            table_name: Table name to check
            
        Returns:
            True if table exists, False otherwise
        """
        return table_name in self.get_tables()
    
    def get_row_count(self, table_name: str) -> int:
        """
        Get number of rows in a table.
        
        Args:
            table_name: Table name
            
        Returns:
            Row count
        """
        result = self.query_df(f"SELECT COUNT(*) as cnt FROM {table_name}")
        return int(result['cnt'].iloc[0])
    
    def sample_rows(self, table_name: str, n: int = 5) -> pd.DataFrame:
        """
        Get sample rows from a table.
        
        Args:
            table_name: Table name
            n: Number of rows to sample
            
        Returns:
            DataFrame with sample rows
        """
        return self.query_df(f"SELECT * FROM {table_name} LIMIT {n}")
    
    def explain_query(self, query: str) -> str:
        """
        Get query execution plan.
        
        Args:
            query: SQL query to explain
            
        Returns:
            Execution plan as string
        """
        result = self.query_df(f"EXPLAIN {query}")
        return result.to_string()
    
    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

