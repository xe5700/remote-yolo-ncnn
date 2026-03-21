# syntax=docker/dockerfile:1
FROM xe5700/ultralytics_ncnn_vulkan:latest
RUN mkdir -p /server/models
WORKDIR /server
COPY /server/requirements.txt .
RUN pip install -r requirements.txt
ADD ./models /server/models
ADD ./server/ /server
CMD [ "python3", "server.py" ]