"""Streamlit BI Dashboard Application."""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.duck import DuckDBManager
from src.utils.io import PathManager, load_config
from src.dspy_programs.sql_synth import SQLSynthesizer
from src.dspy_programs.insight_writer import InsightWriter
import dspy


# Page config
st.set_page_config(
    page_title="BI-DSPy-GEPA Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def get_db():
    """Get cached DuckDB connection."""
    config = load_config()
    db_path = config['project'].get('warehouse_path', 'data/warehouse/bi.duckdb')
    return DuckDBManager(db_path)


@st.cache_resource
def get_schema():
    config = load_config()
    return config.get('schema', {})


@st.cache_resource
def get_dspy_lm(model_name: str, api_key: str):
    """Get cached DSPy LM instance (thread-safe)."""
    return dspy.LM(model=model_name, api_key=api_key, max_tokens=1024, temperature=0.1)


def configure_dspy_safe(model_name: str, api_key: str):
    """Safely configure DSPy for Streamlit's multi-threading environment."""
    try:
        lm = get_dspy_lm(model_name, api_key)
        dspy.configure(lm=lm)
        return True
    except RuntimeError as e:
        # DSPy already configured in another thread
        if "thread" in str(e).lower() or "configured" in str(e).lower():
            # Silently use existing configuration
            return False
        else:
            raise
    except Exception as e:
        st.error(f"Error configuring DSPy: {e}")
        return False


def main():
    """Main dashboard application."""
    st.title("📊 BI-DSPy-GEPA Dashboard")
    st.markdown("**Business Intelligence powered by DSPy + GEPA optimization**")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["📈 KPIs", "📊 Trends", "🔍 Ask a Question", "🚨 Anomalies", "🔬 Audit"]
    )
    
    db = get_db()
    
    if page == "📈 KPIs":
        show_kpis_page(db)
    elif page == "📊 Trends":
        show_trends_page(db)
    elif page == "🔍 Ask a Question":
        show_nl_query_page(db)
    elif page == "🚨 Anomalies":
        show_anomalies_page(db)
    elif page == "🔬 Audit":
        show_audit_page(db)


def show_kpis_page(db: DuckDBManager):
    """Display KPIs page."""
    st.header("📈 Key Performance Indicators")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        date_range = st.selectbox("Date Range", ["Last 7 Days", "Last 30 Days", "Last 90 Days", "All Time"])
    
    # Map date range to days
    days_map = {"Last 7 Days": 7, "Last 30 Days": 30, "Last 90 Days": 90, "All Time": 10000}
    days = days_map[date_range]
    
    # Query KPIs
    query = f"""
    SELECT 
        SUM(revenue) as total_revenue,
        COUNT(DISTINCT order_id) as total_orders,
        COUNT(DISTINCT customer_id) as total_customers,
        SUM(revenue) / COUNT(DISTINCT order_id) as aov,
        SUM(gross_profit) as total_profit,
        AVG(profit_margin) as avg_margin
    FROM fact_orders
    WHERE order_date >= CURRENT_DATE - INTERVAL '{days} days'
    """
    
    try:
        kpis = db.query_df(query)
        
        if not kpis.empty:
            row = kpis.iloc[0]
            
            # Display KPI cards
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Total Revenue",
                    f"${row['total_revenue']:,.2f}",
                    help="Sum of all revenue in selected period"
                )
            
            with col2:
                st.metric(
                    "Total Orders",
                    f"{int(row['total_orders']):,}",
                    help="Number of orders"
                )
            
            with col3:
                st.metric(
                    "Total Customers",
                    f"{int(row['total_customers']):,}",
                    help="Unique customers"
                )
            
            col4, col5, col6 = st.columns(3)
            
            with col4:
                st.metric(
                    "Average Order Value",
                    f"${row['aov']:,.2f}",
                    help="Revenue per order"
                )
            
            with col5:
                st.metric(
                    "Total Profit",
                    f"${row['total_profit']:,.2f}",
                    help="Gross profit"
                )
            
            with col6:
                st.metric(
                    "Avg Margin",
                    f"{row['avg_margin']:.1f}%",
                    help="Average profit margin"
                )
    
    except Exception as e:
        st.error(f"Error loading KPIs: {e}")


def show_trends_page(db: DuckDBManager):
    """Display trends page."""
    st.header("📊 Revenue Trends & Analytics")
    
    # Trend selector
    trend_type = st.selectbox(
        "Select Trend",
        ["Daily Revenue", "Monthly Revenue", "Product Performance", "Customer Segments"],
        help="Choose a trend analysis to display"
    )
    
    try:
        if trend_type == "Daily Revenue":
            query = """
            SELECT 
                order_date,
                SUM(revenue) as revenue,
                COUNT(DISTINCT order_id) as orders,
                SUM(gross_profit) as profit
            FROM fact_orders
            WHERE order_date >= CURRENT_DATE - INTERVAL '90 days'
            GROUP BY order_date
            ORDER BY order_date
            """
            
            data = db.query_df(query)
            
            if not data.empty:
                # Chart
                fig = px.line(data, x='order_date', y='revenue', 
                             title='Daily Revenue (Last 90 Days)',
                             labels={'revenue': 'Revenue ($)', 'order_date': 'Date'},
                             markers=True)
                fig.update_traces(line_color='#1f77b4', line_width=2)
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg Daily Revenue", f"${data['revenue'].mean():,.2f}")
                with col2:
                    st.metric("Total Orders", f"{int(data['orders'].sum()):,}")
                with col3:
                    st.metric("Total Profit", f"${data['profit'].sum():,.2f}")
                
                # Data table
                with st.expander("📋 View Data Table"):
                    st.dataframe(data.tail(10), use_container_width=True)
            else:
                st.warning("No data available for the last 90 days")
                
        elif trend_type == "Monthly Revenue":
            query = """
            SELECT 
                DATE_TRUNC('month', order_date) as month,
                SUM(revenue) as revenue,
                COUNT(DISTINCT order_id) as orders,
                COUNT(DISTINCT customer_id) as customers,
                SUM(gross_profit) as profit,
                AVG(profit_margin) as avg_margin
            FROM fact_orders
            GROUP BY month
            ORDER BY month
            """
            
            data = db.query_df(query)
            
            if not data.empty:
                # Chart
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=data['month'], 
                    y=data['revenue'], 
                    name='Revenue',
                    marker_color='#1f77b4'
                ))
                fig.update_layout(
                    title='Monthly Revenue Trend',
                    xaxis_title='Month',
                    yaxis_title='Revenue ($)',
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Growth analysis
                if len(data) > 1:
                    latest = data.iloc[-1]['revenue']
                    previous = data.iloc[-2]['revenue']
                    growth = ((latest - previous) / previous) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Latest Month Revenue", f"${latest:,.2f}")
                    with col2:
                        st.metric("Previous Month", f"${previous:,.2f}")
                    with col3:
                        st.metric("Month-over-Month Growth", f"{growth:+.1f}%")
                
                # Data table
                with st.expander("📋 View Data Table"):
                    st.dataframe(data, use_container_width=True)
            else:
                st.warning("No monthly data available")
        
        elif trend_type == "Product Performance":
            query = """
            SELECT 
                product_name,
                SUM(revenue) as total_revenue,
                SUM(quantity) as total_quantity,
                SUM(gross_profit) as total_profit,
                AVG(profit_margin) as avg_margin,
                COUNT(DISTINCT customer_id) as unique_customers
            FROM fact_orders
            GROUP BY product_name
            ORDER BY total_revenue DESC
            LIMIT 15
            """
            
            data = db.query_df(query)
            
            if not data.empty:
                # Top products chart
                fig = px.bar(
                    data.head(10), 
                    x='total_revenue', 
                    y='product_name',
                    orientation='h',
                    title='Top 10 Products by Revenue',
                    labels={'total_revenue': 'Revenue ($)', 'product_name': 'Product'},
                    color='total_revenue',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Top Product", data.iloc[0]['product_name'])
                with col2:
                    st.metric("Top Product Revenue", f"${data.iloc[0]['total_revenue']:,.2f}")
                with col3:
                    st.metric("Top Product Margin", f"{data.iloc[0]['avg_margin']:.1f}%")
                
                # Data table
                with st.expander("📋 View Full Product Data"):
                    st.dataframe(data, use_container_width=True)
            else:
                st.warning("No product data available")
        
        elif trend_type == "Customer Segments":
            query = """
            SELECT 
                customer_name,
                SUM(revenue) as total_revenue,
                COUNT(DISTINCT order_id) as total_orders,
                SUM(revenue) / COUNT(DISTINCT order_id) as avg_order_value,
                SUM(gross_profit) as total_profit
            FROM fact_orders
            GROUP BY customer_name
            ORDER BY total_revenue DESC
            LIMIT 20
            """
            
            data = db.query_df(query)
            
            if not data.empty:
                # Top customers chart
                fig = px.bar(
                    data.head(10),
                    x='customer_name',
                    y='total_revenue',
                    title='Top 10 Customers by Revenue',
                    labels={'total_revenue': 'Revenue ($)', 'customer_name': 'Customer'},
                    color='total_revenue',
                    color_continuous_scale='Greens'
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
                
                # Customer insights
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Top Customer", data.iloc[0]['customer_name'])
                with col2:
                    st.metric("Their Revenue", f"${data.iloc[0]['total_revenue']:,.2f}")
                with col3:
                    st.metric("Their Orders", f"{int(data.iloc[0]['total_orders']):,}")
                with col4:
                    st.metric("Their AOV", f"${data.iloc[0]['avg_order_value']:,.2f}")
                
                # Pareto analysis
                data['cumulative_revenue'] = data['total_revenue'].cumsum()
                total_revenue = data['total_revenue'].sum()
                data['cumulative_pct'] = (data['cumulative_revenue'] / total_revenue) * 100
                
                top_20_pct = data[data['cumulative_pct'] <= 80]
                st.info(f"📊 Pareto Analysis: Top {len(top_20_pct)} customers ({len(top_20_pct)/len(data)*100:.1f}%) generate 80% of revenue")
                
                # Data table
                with st.expander("📋 View Customer Data"):
                    st.dataframe(data[['customer_name', 'total_revenue', 'total_orders', 'avg_order_value']], 
                               use_container_width=True)
            else:
                st.warning("No customer data available")
            
    except Exception as e:
        st.error(f"Error loading trends: {e}")
        import traceback
        with st.expander("🔍 Error Details"):
            st.code(traceback.format_exc())


def show_nl_query_page(db: DuckDBManager):
    """Natural language to SQL query page with DSPy+GEPA optimization."""
    st.header("🔍 Ask a Question")
    st.markdown("Ask questions in natural language and get SQL-powered answers using **DSPy + GEPA optimized prompts**")
    
    # Input
    question = st.text_area(
        "Your Question:",
        placeholder="e.g., What are the top 10 customers by revenue?",
        height=100,
        help="Ask natural language questions about your data"
    )
    
    # Advanced options (collapsed)
    with st.expander("⚙️ Advanced Options"):
        use_optimization = st.checkbox("Use GEPA-optimized prompts", value=True, 
                                       help="Uses best genome from GEPA optimization if available")
        show_reasoning = st.checkbox("Show reasoning steps", value=False,
                                     help="Display chain-of-thought reasoning")
    
    if st.button("Get Answer", type="primary"):
        if not question:
            st.warning("Please enter a question")
            return
        
        with st.spinner("🤖 Generating SQL with DSPy+GEPA..."):
            try:
                load_dotenv()
                schema = get_schema()
                
                # Check for API key
                api_key = os.getenv('MISTRAL_API_KEY') or os.getenv('OPENAI_API_KEY') or os.getenv('GOOGLE_API_KEY')
                
                if not api_key:
                    st.error("❌ No API key found!")
                    st.info("""
                    Please configure an API key in your `.env` file:
                    - **MISTRAL_API_KEY** (Recommended, $0.001/1K tokens)
                    - Or OPENAI_API_KEY
                    - Or GOOGLE_API_KEY
                    
                    See README.md for setup instructions.
                    """)
                    return
                
                # Determine model
                if os.getenv('MISTRAL_API_KEY'):
                    model_name = 'mistral/mistral-small-latest'
                    provider = "Mistral"
                elif os.getenv('OPENAI_API_KEY'):
                    model_name = 'gpt-4o-mini'
                    provider = "OpenAI"
                else:
                    model_name = 'gemini-1.5-flash'
                    provider = "Gemini"
                
                # Configure DSPy (thread-safe)
                configured = configure_dspy_safe(model_name, api_key)
                
                # Load GEPA-optimized prompts if available
                optimized_prompts = None
                if use_optimization:
                    try:
                        results_path = Path("eval/results/gepa_real_results.json")
                        if results_path.exists():
                            with open(results_path, 'r') as f:
                                gepa_results = json.load(f)
                                # Extract system prompts from best genome
                                best_genome = gepa_results.get('best_genome', {})
                                components = best_genome.get('components', [])
                                optimized_prompts = {
                                    c['name']: c['text'] 
                                    for c in components 
                                    if c.get('component_type') == 'system'
                                }
                                if optimized_prompts:
                                    st.success(f"✅ Using GEPA-optimized prompts from best genome")
                    except:
                        pass  # Fall back to default prompts
                
                # Initialize SQL Synthesizer with optimization
                synthesizer = SQLSynthesizer(schema_context=schema)
                
                # Generate SQL
                st.info(f"🤖 Using {provider} ({model_name})")
                
                result = synthesizer(task=question)
                
                generated_sql = result.sql.strip()
                
                # Clean up SQL (remove markdown formatting)
                if '```sql' in generated_sql:
                    generated_sql = generated_sql.split('```sql')[1].split('```')[0].strip()
                elif '```' in generated_sql:
                    generated_sql = generated_sql.split('```')[1].split('```')[0].strip()
                
                # Display generated SQL
                st.subheader("📝 Generated SQL")
                st.code(generated_sql, language='sql')
                
                # Show reasoning if requested
                if show_reasoning and hasattr(result, 'rationale'):
                    with st.expander("🧠 Reasoning Steps"):
                        st.write(result.rationale)
                
                # Execute SQL
                st.subheader("📊 Results")
                try:
                    query_result = db.query_df(generated_sql)
                    
                    if query_result.empty:
                        st.warning("Query returned no results")
                    else:
                        # Display results
                        st.dataframe(query_result, use_container_width=True)
                        st.caption(f"Returned {len(query_result)} row(s)")
                        
                        # Export options
                        col1, col2 = st.columns(2)
                        with col1:
                            csv = query_result.to_csv(index=False)
                            st.download_button(
                                "📥 Download CSV",
                                csv,
                                f"query_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                "text/csv",
                                help="Download results as CSV file"
                            )
                        with col2:
                            st.download_button(
                                "📥 Download SQL",
                                generated_sql,
                                f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sql",
                                "text/plain",
                                help="Download SQL query"
                            )
                        
                        # Auto-visualize if appropriate
                        if len(query_result) > 0 and len(query_result.columns) >= 2:
                            with st.expander("📈 Quick Visualization"):
                                viz_type = st.selectbox("Chart Type", ["Bar Chart", "Line Chart", "Scatter Plot", "None"])
                                
                                if viz_type != "None" and len(query_result.columns) >= 2:
                                    x_col = st.selectbox("X Axis", query_result.columns, index=0)
                                    y_col = st.selectbox("Y Axis", query_result.columns, index=1)
                                    
                                    if viz_type == "Bar Chart":
                                        fig = px.bar(query_result, x=x_col, y=y_col)
                                    elif viz_type == "Line Chart":
                                        fig = px.line(query_result, x=x_col, y=y_col)
                                    else:
                                        fig = px.scatter(query_result, x=x_col, y=y_col)
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                
                except Exception as exec_error:
                    st.error(f"❌ SQL Execution Error: {exec_error}")
                    st.warning("The generated SQL may need manual adjustment. You can copy and modify it above.")
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
                import traceback
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())


def show_anomalies_page(db: DuckDBManager):
    """Display anomalies and alerts page."""
    st.header("🚨 Anomalies & Alerts")
    
    st.markdown("Automated anomaly detection on key metrics")
    
    # Get recent data for anomaly detection
    query = """
    SELECT 
        order_date,
        SUM(revenue) as daily_revenue,
        COUNT(DISTINCT order_id) as daily_orders,
        AVG(revenue) as avg_order_value
    FROM fact_orders
    WHERE order_date >= CURRENT_DATE - INTERVAL '90 days'
    GROUP BY order_date
    ORDER BY order_date
    """
    
    try:
        data = db.query_df(query)
        
        if not data.empty and len(data) > 10:
            # Simple anomaly detection using z-score
            from scipy import stats
            
            data['revenue_zscore'] = stats.zscore(data['daily_revenue'])
            anomalies = data[abs(data['revenue_zscore']) > 2]
            
            if not anomalies.empty:
                st.warning(f"⚠️ Detected {len(anomalies)} anomalous day(s)")
                
                # Display anomalies
                st.dataframe(
                    anomalies[['order_date', 'daily_revenue', 'daily_orders', 'revenue_zscore']],
                    use_container_width=True
                )
                
                # Visualize
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=data['order_date'],
                    y=data['daily_revenue'],
                    mode='lines',
                    name='Revenue',
                    line=dict(color='blue')
                ))
                
                fig.add_trace(go.Scatter(
                    x=anomalies['order_date'],
                    y=anomalies['daily_revenue'],
                    mode='markers',
                    name='Anomalies',
                    marker=dict(color='red', size=10)
                ))
                
                fig.update_layout(
                    title='Revenue with Anomalies Highlighted',
                    xaxis_title='Date',
                    yaxis_title='Revenue ($)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ No anomalies detected in the last 90 days")
        else:
            st.info("Not enough data for anomaly detection (need 10+ days)")
            
    except Exception as e:
        st.error(f"Error detecting anomalies: {e}")


def show_audit_page(db: DuckDBManager):
    """Display audit and lineage page."""
    st.header("🔬 Audit & Lineage")
    
    st.markdown("Data quality checks and query audit trail")
    
    # Show available tables
    st.subheader("Data Warehouse Tables")
    
    tables = db.get_tables()
    
    if tables:
        for table in tables:
            with st.expander(f"📋 {table}"):
                # Show row count
                count_query = f"SELECT COUNT(*) as count FROM {table}"
                count_result = db.query_df(count_query)
                
                if not count_result.empty:
                    row_count = count_result.iloc[0]['count']
                    st.metric("Total Rows", f"{row_count:,}")
                
                # Show sample data
                sample_query = f"SELECT * FROM {table} LIMIT 5"
                sample_data = db.query_df(sample_query)
                
                if not sample_data.empty:
                    st.dataframe(sample_data, use_container_width=True)
    else:
        st.warning("No tables found in warehouse")


if __name__ == "__main__":
    main()
