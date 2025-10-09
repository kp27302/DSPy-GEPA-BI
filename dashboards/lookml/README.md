# LookML Templates

This directory contains LookML templates for Looker/Google Cloud BI integration.

## Future Enhancement

LookML export is a planned feature. Contributions welcome!

## Proposed Structure

```lookml
view: fact_orders {
  sql_table_name: fact_orders ;;
  
  dimension: order_id {
    type: number
    sql: ${TABLE}.order_id ;;
    primary_key: yes
  }
  
  dimension_group: order {
    type: time
    timeframes: [date, week, month, quarter, year]
    sql: ${TABLE}.order_date ;;
  }
  
  measure: total_revenue {
    type: sum
    sql: ${TABLE}.revenue ;;
    value_format_name: usd
  }
  
  measure: order_count {
    type: count_distinct
    sql: ${TABLE}.order_id ;;
  }
}
```

## Contributing

To add LookML generation:
1. Create `src/metrics/lookml_writer.py`
2. Parse `kpis.yaml` and generate LookML
3. Add CLI command: `make export-lookml`

