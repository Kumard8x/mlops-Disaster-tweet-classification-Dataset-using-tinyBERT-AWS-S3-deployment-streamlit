from sys import prefix
import streamlit as st
import boto3
import os
import torch
from transformers import pipeline

bucket_name="mlops-30-102025"


local_path='tinybert-disaster-tweet'
s3_prefix = 'ml-model/tinybert-sentiment-analysis'

s3=boto3.client('s3', region_name='us-east-1')



#download model from s3 bucket.
def download_dir(local_path, s3_prefix):
    os.makedirs(local_path, exist_ok=True)
    paginator=s3.get_paginator('list_objects_v2')
    
    for result in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        if 'Contents' in result:
            for key in result['Contents']:
                s3_key=key['Key'] #type:ignore
                
                relative_path=os.path.relpath(s3_key, s3_prefix)
                local_file=os.path.join(local_path, relative_path)
                
                os.makedirs(os.path.dirname(local_file), exist_ok=True)
                
                s3.download_file(bucket_name, s3_key, local_file )
                
st.title("Machine learning model deploy on server.")    
st.markdown("### Tweet Disaster analysis using TinyBERT transofer.")

button = st.button("Click here to download model")
if button:
    try: 
        if not os.listdir(local_path):
            with st.spinner("Please wail .... Downloading"):
                download_dir(local_path, s3_prefix)
                  
    except Exception as e:
        st.error(f'Facing Model download problem from AWS S3 server side.\n Sorry Can not use this model without dowload. \n {e}')
        
text = st.text_area("Enter your Tweet here")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

predict = st.button('Predict')


if os.listdir(local_path):
    classifier = pipeline('text-classification', model='tinybert-disaster-tweet', device=device)
    if predict:
        with st.spinner("wait.."):
            output = classifier(text)
        st.write(output)
else:
    st.warning("Model is not availabel. Please first Download the model then use it.")
