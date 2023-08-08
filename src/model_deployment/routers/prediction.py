from fastapi import APIRouter
from typing import List
from src.model_deployment.models.input_models import PredictionInput
from src.model_deployment.models.output_models import PredictionOutput
from src.model_deployment.utils.nlp_utils import process_tokenize_input, get_ort_session, run_inference


router = APIRouter()


@router.post("/send_data", response_model=List[PredictionOutput])
async def send_data(prediction_input: PredictionInput):
    if isinstance(prediction_input.sequences, str):
        prediction_input.sequences = [prediction_input.sequences, ]

    ort_inputs = process_tokenize_input(prediction_input.sequences, prediction_input.preprocess)
    session = get_ort_session()

    output_classes, output_confidences, output_logits = run_inference(session, ort_inputs )
    outputs = [PredictionOutput(predicted_class=output_class, confidence=conf, logits=logits) for output_class, conf, logits in zip(output_classes, output_confidences, output_logits)]
    return outputs