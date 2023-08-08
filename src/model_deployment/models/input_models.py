from pydantic import BaseModel
from typing import List, Optional

class PredictionInput(BaseModel):
    sequences: List[str] | str
    preprocess: Optional[bool] = True