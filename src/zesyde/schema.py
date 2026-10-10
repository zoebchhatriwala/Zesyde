"""Core data schemas and serialization for Zesyde decision instances.

Defines the contract for decision questions across three primitives:
- `choice`: dynamic discrete classification over N options.
- `noul`: boolean / calibrated probability of truth (options: [False, True]).
- `score`: ordinal rating (e.g. 1 to K) with inherent order.
"""
from enum import Enum
from typing import List, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator

class QuestionType(str, Enum):
    CHOICE = 'choice'
    NOUL = 'noul'
    SCORE = 'score'

class DecisionExample(BaseModel):
    """A single decision instance conforming to the Zesyde spec."""
    id: str = Field(..., description='Unique identifier for the example')
    context: str = Field(..., description='Context, state, or conversation prefix')
    question: str = Field(..., description='The decision question being asked')
    question_type: QuestionType = Field(..., description="Primitive: 'choice', 'noul', or 'score'")
    options: List[str] = Field(..., description='Candidate labels or ordinal levels for this decision instance')
    target: int = Field(..., description='Ground truth target index pointing into `options` (0-indexed)')
    teacher_probs: Optional[List[float]] = Field(None, description='Soft probability distribution from teacher model for distillation')
    metadata: Optional[dict] = Field(default_factory=dict, description='Arbitrary task metadata (e.g., domain, benchmark name, split)')

    @field_validator('options')
    @classmethod
    def validate_options(cls, v: List[str]) -> List[str]:
        if len(v) < 2:
            raise ValueError(f'A decision must have at least 2 options, got {len(v)}')
        if len(v) > 20:
            raise ValueError(f'Option count {len(v)} exceeds maximum supported budget (~20 options)')
        return v

    @model_validator(mode='after')
    def validate_target_and_probs(self) -> 'DecisionExample':
        num_options = len(self.options)
        if not (0 <= self.target < num_options):
            raise ValueError(f'Target index {self.target} is out of bounds for options of length {num_options}')
        
        if self.question_type == QuestionType.NOUL:
            if num_options != 2:
                raise ValueError(f"Primitive 'noul' must have exactly 2 options (typically ['false', 'true'] or ['no', 'yes']), got {num_options}")
                
        if self.teacher_probs is not None:
            if len(self.teacher_probs) != num_options:
                raise ValueError(f'Teacher distribution length ({len(self.teacher_probs)}) must match option count ({num_options})')
            total = sum(self.teacher_probs)
            if not (0.98 <= total <= 1.02):
                raise ValueError(f'Teacher distribution must sum to approximately 1.0, got {total}')
                
        return self

class DecisionBatch(BaseModel):
    """Batch of decision instances."""
    examples: List[DecisionExample]
