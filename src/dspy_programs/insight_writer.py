"""DSPy program for generating narrative insights from BI data."""

import dspy
from typing import Dict, Any, List, Optional
import pandas as pd
from dataclasses import dataclass


@dataclass
class InsightResult:
    """Result of insight generation."""
    summary: str
    key_findings: List[str]
    recommendations: List[str]
    suggested_queries: List[str]


class GenerateInsights(dspy.Signature):
    """Generate executive insights from BI data."""
    
    kpi_data = dspy.InputField(desc="KPI values and trends over time")
    context = dspy.InputField(desc="Business context and previous period comparisons")
    anomalies = dspy.InputField(desc="Detected anomalies or significant changes")
    
    summary = dspy.OutputField(desc="Executive summary paragraph")
    key_findings = dspy.OutputField(desc="Bullet list of 3-5 key findings")
    recommendations = dspy.OutputField(desc="Actionable recommendations")


class SuggestNextQueries(dspy.Signature):
    """Suggest follow-up analytical queries based on insights."""
    
    insights = dspy.InputField(desc="Generated insights and findings")
    available_dimensions = dspy.InputField(desc="Available dimensions for drill-down")
    
    queries = dspy.OutputField(desc="List of suggested follow-up queries in natural language")


class InsightWriter(dspy.Module):
    """
    DSPy module for writing narrative insights from BI metrics.
    Generates executive summaries, key findings, and suggested next queries.
    """
    
    def __init__(self):
        """Initialize insight writer."""
        super().__init__()
        
        self.generate_insights = dspy.ChainOfThought(GenerateInsights)
        self.suggest_queries = dspy.ChainOfThought(SuggestNextQueries)
    
    def _format_kpi_data(self, kpi_df: pd.DataFrame) -> str:
        """Format KPI DataFrame for LLM context."""
        if kpi_df.empty:
            return "No KPI data available."
        
        # Get summary statistics
        summary = f"KPI Data Summary ({len(kpi_df)} rows):\n\n"
        
        # Show recent values
        if len(kpi_df) > 0:
            summary += "Recent Values:\n"
            summary += kpi_df.tail(10).to_string(index=False)
            summary += "\n\n"
        
        # Compute trends if possible
        if 'revenue' in kpi_df.columns and len(kpi_df) > 1:
            latest = kpi_df['revenue'].iloc[-1]
            previous = kpi_df['revenue'].iloc[-2]
            change = ((latest - previous) / previous * 100) if previous != 0 else 0
            summary += f"Revenue trend: {change:+.1f}% vs previous period\n"
        
        return summary
    
    def _detect_anomalies(self, kpi_df: pd.DataFrame, 
                         threshold_std: float = 2.0) -> List[str]:
        """Simple anomaly detection using standard deviation."""
        anomalies = []
        
        for col in kpi_df.select_dtypes(include=['float64', 'int64']).columns:
            if col == 'order_date' or col.endswith('_date'):
                continue
            
            mean = kpi_df[col].mean()
            std = kpi_df[col].std()
            
            if std == 0:
                continue
            
            outliers = kpi_df[abs(kpi_df[col] - mean) > threshold_std * std]
            
            if len(outliers) > 0:
                anomalies.append(
                    f"{col}: {len(outliers)} outlier(s) detected "
                    f"(>{threshold_std}σ from mean={mean:.2f})"
                )
        
        return anomalies if anomalies else ["No significant anomalies detected"]
    
    def forward(self, kpi_data: pd.DataFrame, 
                context: Optional[str] = None,
                dimensions: Optional[List[str]] = None) -> InsightResult:
        """
        Generate insights from KPI data.
        
        Args:
            kpi_data: DataFrame with KPI metrics
            context: Optional business context
            dimensions: Available dimensions for drill-down
            
        Returns:
            InsightResult with summary, findings, and recommendations
        """
        # Format inputs
        kpi_str = self._format_kpi_data(kpi_data)
        context = context or "General business performance review"
        anomalies = self._detect_anomalies(kpi_data)
        anomalies_str = "\n".join(anomalies)
        
        # Generate insights
        result = self.generate_insights(
            kpi_data=kpi_str,
            context=context,
            anomalies=anomalies_str
        )
        
        # Parse key findings
        findings = result.key_findings.strip().split('\n')
        findings = [f.strip('- •*') for f in findings if f.strip()]
        
        # Parse recommendations
        recommendations = result.recommendations.strip().split('\n')
        recommendations = [r.strip('- •*') for r in recommendations if r.strip()]
        
        # Generate suggested queries
        dimensions = dimensions or ['region', 'segment', 'category']
        query_result = self.suggest_queries(
            insights=result.summary + "\n" + result.key_findings,
            available_dimensions=str(dimensions)
        )
        
        suggested_queries = query_result.queries.strip().split('\n')
        suggested_queries = [q.strip('- •*') for q in suggested_queries if q.strip()]
        
        return InsightResult(
            summary=result.summary.strip(),
            key_findings=findings[:5],  # Limit to 5
            recommendations=recommendations[:3],  # Limit to 3
            suggested_queries=suggested_queries[:5]  # Limit to 5
        )


def generate_insights(kpi_data: pd.DataFrame, 
                     model: str = "openai/gpt-4o-mini") -> InsightResult:
    """
    Standalone function to generate insights.
    
    Args:
        kpi_data: DataFrame with KPI metrics
        model: LLM model
        
    Returns:
        InsightResult
    """
    lm = dspy.OpenAI(model=model, max_tokens=1500, temperature=0.3)
    dspy.settings.configure(lm=lm)
    
    writer = InsightWriter()
    return writer(kpi_data=kpi_data)

