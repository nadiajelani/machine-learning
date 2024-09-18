#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import boto3
import base64
from sagemaker.serializers import IdentitySerializer

# Lambda 1: serializeImageData
s3 = boto3.client('s3')

def serializeImageData(event, context):
    """A function to serialize target data from S3"""
    key = event["s3_key"]
    bucket = event["s3_bucket"]

    # Download the data from S3 to /tmp/image.png
    s3.download_file(bucket, key, "/tmp/image.png")

    # Read the data from a file and encode it
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')

    # Pass the serialized data back to Step Function
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }

# Lambda 2: imageClassifier
ENDPOINT = "your-sagemaker-endpoint-name"
runtime = boto3.client('runtime.sagemaker')

def imageClassifier(event, context):
    """A function to classify images"""
    image = base64.b64decode(event["image_data"])

    # Instantiate the Predictor
    predictor = runtime.invoke_endpoint(
        EndpointName=ENDPOINT,
        ContentType="image/png",
        Body=image
    )

    inferences = predictor['Body'].read()

    # We return the inferences back to the Step Function
    event["inferences"] = json.loads(inferences.decode('utf-8'))
    return {
        'statusCode': 200,
        'body': event
    }

# Lambda 3: thresholdFilter
THRESHOLD = 0.93

def thresholdFilter(event, context):
    """A function to filter low-confidence inferences"""
    inferences = event["inferences"]

    # Check if any of the inferences meet the threshold
    meets_threshold = any(float(inf) > THRESHOLD for inf in inferences)

    # If the threshold is met, pass the data, else raise an error
    if meets_threshold:
        return {
            'statusCode': 200,
            'body': event
        }
    else:
        raise Exception("THRESHOLD_CONFIDENCE_NOT_MET")

