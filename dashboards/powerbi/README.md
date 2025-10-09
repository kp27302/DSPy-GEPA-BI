# Power BI Integration

This directory is for Power BI exports and integration.

## DAX Measures

Generate DAX measures for Power BI:

```bash
python src/metrics/dax_writer.py
```

Output: `measures.txt` with all KPI definitions in DAX format.

## Connecting Power BI to DuckDB

### Option 1: Export to Parquet

1. Run ETL: `make etl`
2. Find Parquet files in `data/warehouse/*.parquet`
3. In Power BI: Get Data → Parquet
4. Select files to import

### Option 2: Export to CSV

In Streamlit dashboard:
1. Run queries
2. Use "Download CSV" button
3. Import CSV into Power BI

### Option 3: ODBC (Advanced)

1. Install DuckDB ODBC driver
2. Create ODBC connection to `data/warehouse/bi.duckdb`
3. Use Power BI → Get Data → ODBC

## Sample Dashboard Structure

Recommended pages:
- Executive Summary (KPIs)
- Revenue Analysis (trends, regions)
- Customer Insights (segmentation, cohorts)
- Product Performance (top products, categories)
- Anomaly Detection (outliers, alerts)

## Placeholder File

`BI_Project.pbix` is a placeholder. Create your actual Power BI report and save it here.

