from pydantic import BaseModel
from typing import List


class PredictionOutput(BaseModel):
    predicted_class: int
    confidence: float
    logits: List[float]
