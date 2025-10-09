"""Evolvable text components (prompt genomes)."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import hashlib
import json


@dataclass
class TextComponent:
    """
    A single evolvable text component (gene).
    Could be a system prompt, few-shot example, tool hint, etc.
    """
    name: str
    text: str
    component_type: str  # 'system', 'fewshot', 'tool_hint', 'guard'
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        """Hash based on content."""
        content = f"{self.name}:{self.text}"
        return int(hashlib.md5(content.encode()).hexdigest()[:8], 16)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'text': self.text,
            'component_type': self.component_type,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TextComponent':
        """Create from dictionary."""
        return cls(
            name=data['name'],
            text=data['text'],
            component_type=data['component_type'],
            metadata=data.get('metadata', {})
        )


@dataclass
class PromptGenome:
    """
    A complete prompt policy as a genome.
    Comprises multiple text components that can be evolved.
    """
    genome_id: str
    components: List[TextComponent]
    metadata: Dict[str, Any] = field(default_factory=dict)
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    
    def __hash__(self):
        """Hash based on components."""
        component_hashes = tuple(sorted(hash(c) for c in self.components))
        return hash((self.genome_id, component_hashes))
    
    def get_component(self, name: str) -> Optional[TextComponent]:
        """Get component by name."""
        for comp in self.components:
            if comp.name == name:
                return comp
        return None
    
    def get_components_by_type(self, comp_type: str) -> List[TextComponent]:
        """Get all components of a given type."""
        return [c for c in self.components if c.component_type == comp_type]
    
    def to_prompt_dict(self) -> Dict[str, Any]:
        """
        Convert genome to prompt configuration for DSPy.
        
        Returns:
            Dictionary with system prompts, few-shots, hints
        """
        prompt_config = {
            'system': [],
            'fewshots': [],
            'tool_hints': [],
            'guards': []
        }
        
        for comp in self.components:
            if comp.component_type == 'system':
                prompt_config['system'].append(comp.text)
            elif comp.component_type == 'fewshot':
                prompt_config['fewshots'].append(comp.text)
            elif comp.component_type == 'tool_hint':
                prompt_config['tool_hints'].append(comp.text)
            elif comp.component_type == 'guard':
                prompt_config['guards'].append(comp.text)
        
        return prompt_config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'genome_id': self.genome_id,
            'components': [c.to_dict() for c in self.components],
            'metadata': self.metadata,
            'generation': self.generation,
            'parent_ids': self.parent_ids
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptGenome':
        """Create from dictionary."""
        return cls(
            genome_id=data['genome_id'],
            components=[TextComponent.from_dict(c) for c in data['components']],
            metadata=data.get('metadata', {}),
            generation=data.get('generation', 0),
            parent_ids=data.get('parent_ids', [])
        )
    
    def save_json(self, path: str):
        """Save genome to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_json(cls, path: str) -> 'PromptGenome':
        """Load genome from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


def create_baseline_genome() -> PromptGenome:
    """
    Create a baseline prompt genome for SQL synthesis.
    
    Returns:
        PromptGenome with default components
    """
    components = [
        TextComponent(
            name="system_base",
            text="You are an expert SQL developer. Generate accurate, efficient SQL queries.",
            component_type="system"
        ),
        TextComponent(
            name="schema_instruction",
            text="Always refer to the provided schema. Use proper table and column names.",
            component_type="system"
        ),
        TextComponent(
            name="fewshot_1",
            text="Task: Get total revenue by region\\nSQL: SELECT region, SUM(revenue) FROM fact_orders GROUP BY region",
            component_type="fewshot"
        ),
        TextComponent(
            name="tool_hint_join",
            text="Prefer INNER JOIN on primary/foreign keys for referential integrity.",
            component_type="tool_hint"
        ),
        TextComponent(
            name="guard_no_select_star",
            text="Avoid SELECT * - explicitly list columns needed.",
            component_type="guard"
        )
    ]
    
    return PromptGenome(
        genome_id="baseline_v1",
        components=components,
        metadata={'created': 'baseline'}
    )

