FROM bitnami/pytorch
RUN pip install torch===1.5.0 torchvision===0.6.0 
RUN pip install wheel
RUN pip install pandas
RUN pip install matplotlib
