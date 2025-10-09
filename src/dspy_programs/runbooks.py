"""DSPy runbooks for automated remediation and RCA."""

import dspy
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class RunbookResult:
    """Result of runbook execution."""
    diagnosis: str
    root_cause: str
    fix_proposal: str
    confidence: float


class DiagnoseFailure(dspy.Signature):
    """Diagnose test failure and identify root cause."""
    
    test_name = dspy.InputField(desc="Name of failed test")
    error_message = dspy.InputField(desc="Error message or failure details")
    context = dspy.InputField(desc="System context: schema, recent changes, etc.")
    
    diagnosis = dspy.OutputField(desc="Diagnosis of the failure")
    root_cause = dspy.OutputField(desc="Likely root cause")
    fix_proposal = dspy.OutputField(desc="Proposed fix or remediation steps")


class Runbook(dspy.Module):
    """
    DSPy runbook for automated diagnostics and fix proposals.
    When tests fail, suggest corrective actions.
    """
    
    def __init__(self):
        """Initialize runbook."""
        super().__init__()
        self.diagnose = dspy.ChainOfThought(DiagnoseFailure)
    
    def forward(self, test_name: str, error_message: str,
                context: Optional[str] = None) -> RunbookResult:
        """
        Diagnose failure and propose fix.
        
        Args:
            test_name: Failed test name
            error_message: Error details
            context: Optional system context
            
        Returns:
            RunbookResult with diagnosis and fix
        """
        context = context or "No additional context provided"
        
        result = self.diagnose(
            test_name=test_name,
            error_message=error_message,
            context=context
        )
        
        return RunbookResult(
            diagnosis=result.diagnosis.strip(),
            root_cause=result.root_cause.strip(),
            fix_proposal=result.fix_proposal.strip(),
            confidence=0.75  # Could be derived from model confidence
        )

