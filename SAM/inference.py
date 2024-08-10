## inference.py

import json
import base64
import io
from PIL import Image

import torch
from transformers import SamModel, SamProcessor


# model = SamModel.from_pretrained("facebook/sam-vit-huge")
# processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

### better to load model/processor (in model_fn etc) separately for each sagemaker instance rather than globally 
### if mmultiple instances


model = None
processor = None


def model_fn(model_dir, context=None):
    #print(f"model_dir: {model_dir}")
    global model, processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SamModel.from_pretrained(model_dir).to(device)
    processor = SamProcessor.from_pretrained(model_dir)
    return model


def input_fn(input_data, content_type):

    # if content_type == "application/jsonlines":
    #     requests = [json.loads(line) for line in input_data.splitlines()]
    #     processed_inputs_list = []

    #     for request in requests:
    #         image = Image.open(io.BytesIO(base64.b64decode(request['image'])))
    #         input_points = request['input_points']
    #         input_boxes = request.get('input_boxes', None)

    #         processed_inputs = processor(image, input_points=input_points, input_boxes=input_boxes, return_tensors="pt")
    #         processed_inputs_list.append(processed_inputs)
    #     return processed_inputs_list

    if content_type == "application/jsonlines":
        requests = [json.loads(line) for line in input_data.splitlines()]

        images = [Image.open(io.BytesIO(base64.b64decode(request['image']))) for request in requests]
        input_points = [request['input_points'] for request in requests]
        #input_boxes = [request.get('input_boxes', []) for request in requests] 

        #processed_inputs = processor(images, input_points=input_points, input_boxes=input_boxes, return_tensors="pt")
        processed_inputs = processor(images, input_points=input_points, return_tensors="pt")
        return processed_inputs


    else:
        raise ValueError("Unsupported content type: {}".format(content_type))
    

def predict_fn(processed_inputs, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processed_inputs = processed_inputs.to(device)
    with torch.no_grad():    
       masks = processor.image_processor.post_process_masks(
                    model(**processed_inputs).pred_masks.cpu(), processed_inputs["original_sizes"].cpu(), processed_inputs["reshaped_input_sizes"].cpu())


    output_masks = []

    for mask in masks:
        mask = mask[:, 0, :, :].numpy()
        mask = mask.squeeze(0)
        mask_bytes = mask.tobytes()
        output = base64.b64encode(mask_bytes).decode('utf-8')
        output_masks.append(output)
       
    return output_masks


def output_fn(prediction_output, accept):
    if accept == "application/json":
        return json.dumps({"segmentation_masks": prediction_output})
    else:
        raise ValueError(f"Unsupported accept type: {accept}")

