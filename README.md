# Image Classification using AWS SageMaker
Use AWS SageMaker to train a pre-trained model that can perform image classification by using SageMaker profiling, debugger, hyperparameter tuning, and other good ML engineering practices on the CIFAR dataset.

## Project Set Up and Installation
To get started, access AWS through the gateway in the course and open SageMaker Studio. Once you're in SageMaker Studio, download the starter files for this project. You will also need to download the CIFAR dataset or make it available.

## Dataset
The dataset used for this project is CIFAR, which is available in the SageMaker Studio. CIFAR is a widely used dataset for image classification tasks, consisting of 60,000 32x32 color images in 10 classes.

## Access
The CIFAR dataset is already available in SageMaker Studio, so you do not need to upload it to an S3 bucket.

## Hyperparameter Tuning
For this experiment, a pre-trained ResNet18 model was chosen. ResNet is a popular convolutional neural network architecture for image classification tasks, and ResNet18 is a smaller version of the original ResNet model that can be trained more quickly.
The hyperparameters were searched over the following ranges:

```
hyperparameter_ranges = {
    "learning_rate": ContinuousParameter(0.001, 0.1),
    "momentum": ContinuousParameter(0.1, 1),
    "epochs": IntegerParameter(2, 4)
}
```

![screenshots/hyperparam_jobs.png](Hyperperam jobs)

## Training
Several training jobs were run with different combinations of hyperparameters, and the best hyperparameters were chosen based on the validation accuracy, from which a training job was performed.
![screenshots/training_jobs.png](Training jobs)

## Debugging and Profiling
SageMaker's debugger was used to monitor the training process and detect any errors or anomalies. The profiling feature was used to identify any performance bottlenecks in the training process. The profiler report showed that the most time-consuming part of the training process was the forward propagation step, which accounted for about 90% of the total training time.

## Results
The training process achieved a validation accuracy of around 23% with the best hyperparameters. The profiler report showed that the forward propagation step was the most time-consuming, so further optimization could be achieved by improving the efficiency of this step.

The profiler HTML file is included in the submission.

## Model Deployment
The final trained model was deployed as an endpoint in SageMaker. The endpoint can be queried with a sample input image to classify the image into one of the 10 classes.

A screenshot of the deployed active endpoint in SageMaker is not included in the submission due to being too costly to run it online. Hence, we run it offline, and in the notebook we can see the prediction.

## Standout Suggestions
N/A.