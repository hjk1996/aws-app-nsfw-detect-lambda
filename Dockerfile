FROM public.ecr.aws/lambda/python:3.11
ENV TRANSFORMERS_CACHE=/cache_dir
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install transformers
ADD lambda_function.py /var/task/
ADD model /var/task/model
RUN mkdir /cache_dir
CMD ["lambda_function.handler"]
