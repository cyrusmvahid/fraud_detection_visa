
# Excel in tuning models using Amazon LinearLearner Algorithm

# Assumptions and Disclaimers
This blogpost assumes that you have already completed the following tutorials from Amazon SageMaker docuemntation:
- [Setting up](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-set-up.html)
- [Create am Amazon SageMaker Notebook Instance](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-setup-working-env.html)
- I have included 'sagemaker' in the name of my S3 bucket, "cyrusmv-sagemaker-demos' and have chosen to let any SageMaker notebook instance to access any S3 bucket with the term 'sagemaker' included in the name. This is however is not a recommended security option for production and is only useful for simplifying the flow of the blog.
- It is assumed that the reader is familiar with linear regression. If not please read part 1 of this post, Linear Regression and Binary Classification, a Friendly Introduction.



# Hyperparameter Tuning
In the previous part we used default hyperparameters:
u'epochs': u'10', u'init_bias': u'0.0', u'lr_scheduler_factor': u'0.99', u'num_calibration_samples': u'10000000', u'_num_kv_servers': u'auto', u'use_bias': u'true', u'num_point_for_scaler': u'10000', u'_log_level': u'info', u'bias_lr_mult': u'10', u'lr_scheduler_step': u'100', u'init_method': u'uniform', u'init_sigma': u'0.01', u'lr_scheduler_minimum_lr': u'0.00001', **u'target_recall': u'0.8'**, **u'num_models': u'32'**, u'momentum': u'0.0', u'unbias_label': u'auto', u'wd': u'0.0', u'optimizer': u'adam', u'learning_rate': u'auto', u'_kvstore': u'auto', **u'normalize_data': u'true'**, **u'binary_classifier_model_selection_criteria': u'accuracy'**, u'use_lr_scheduler': u'true', **u'target_precision': u'0.8'**, u'force_dense': u'true', u'unbias_data': u'auto', u'init_scale': u'0.07', u'bias_wd_mult': u'0', u'mini_batch_size': u'1000', u'beta_1': u'0.9', u'loss': u'auto', u'beta_2': u'0.999', u'normalize_label': u'auto', u'_num_gpus': u'auto', u'_data_format': u'record', u'positive_example_weight_mult': u'1.0', u'l1': u'0.0'}

Let us highlight a few of these parameters:
- **target_recall and target_precision** are both set to 80%. As we intend to optimize for recall, we change the recall target to 90% and see what accuracy we are going to acheive.
- **normalize_data** is already true.
- **binary_classifier_model_selection_criteria** is accuracy. I will change it to **precision_at_target_recall**. This forces the model to optimize for recall of 90% whatever the accuracy might end up at.
- **num_models** is 32 so we know that HPO is running 32 models in parallel.

for more information on linear learner hyperparmetyers pleaese refer to [Amazon SageMaker Documentaiton](https://docs.aws.amazon.com/sagemaker/latest/dg/ll_hyperparameters.html)



```python
#imports
import boto3 #AWS python SDK for accessing AWS services
import numpy as np #Array libraru with probability and statistics capabilities
import io
import sagemaker.amazon.common as smac # Amazon Sagemaker common library that includes data formats
import sagemaker #sagemaker python sdk
import os
from sagemaker.predictor import csv_serializer, json_deserializer #sagemaker prediction sdk
from sagemaker import get_execution_role

```


```python
bucket = 'cyrusmv-sagemaker-demos'     #replace this with your own bucket 
original_key = 'visa-kaggle/original.csv'    #replace this with your own file inside the bucket
local_pickel_root = '../data/'
dist = 'visa-kaggle/data/'
s3_4_output = 'visa-kaggle/'

files = {}

role = get_execution_role() #this is SageMaker role that would be later used for authorizing SageMaker to access S3
print(role) 

sagemaker_session = sagemaker.Session()
```

    arn:aws:iam::475933981307:role/service-role/AmazonSageMaker-ExecutionRole-20180102T172706


# Downloading Data Files from S3
We iterate over S3 subfolders recursively and when reaching a leaf, we download the file. We also append the location of the file and key to files array, so the code can be generalized based on your folder structure in S3. 

*Disclaimer: The code here is based on [this stackoverflow reference](https://stackoverflow.com/questions/31918960/boto3-to-download-all-files-from-a-s3-bucket) plus exception handling and creating a dictionary of files.*


```python
def download_dir(client, resource, dist, local, bucket):
    paginator = client.get_paginator('list_objects')
    for result in paginator.paginate(Bucket=bucket, Delimiter='/', Prefix=dist):
        if result.get('CommonPrefixes') is not None:
            for subdir in result.get('CommonPrefixes'):
                download_dir(client, resource, subdir.get('Prefix'), local, bucket)
        if result.get('Contents') is not None:
            for file in result.get('Contents'):
                if not os.path.exists(os.path.dirname(local + os.sep + file.get('Key'))):
                    os.makedirs(os.path.dirname(local + os.sep + file.get('Key')))
                print('bucket: {} source file: {}; ==> local: {} \n'.format(bucket, file.get('Key'), local + os.sep + file.get('Key')))
                try:
                    dest = local + os.sep + file.get('Key')
                    key = dest.rsplit('/',1)[-1]
                    key = key.rsplit('.', 1)[0]
                    resource.meta.client.download_file(bucket, file.get('Key'),dest)
                    files[key] = dest
                except (IsADirectoryError, NotADirectoryError):
                    print('WARNING: {}/{} is a directory, skipping download operation'.format(bucket, file.get('Key')))

                    
def _start():
    client = boto3.client('s3')
    resource = boto3.resource('s3')
    download_dir(client, resource, local=local_pickel_root, bucket=bucket, dist=dist)
    print('\ndownload completed.')
    
_start()

files

```

    bucket: cyrusmv-sagemaker-demos source file: visa-kaggle/data/test/val_data.npy; ==> local: ../data//visa-kaggle/data/test/val_data.npy 
    
    bucket: cyrusmv-sagemaker-demos source file: visa-kaggle/data/test/val_label.npy; ==> local: ../data//visa-kaggle/data/test/val_label.npy 
    
    bucket: cyrusmv-sagemaker-demos source file: visa-kaggle/data/train/train_data.npy; ==> local: ../data//visa-kaggle/data/train/train_data.npy 
    
    bucket: cyrusmv-sagemaker-demos source file: visa-kaggle/data/train/train_label.npy; ==> local: ../data//visa-kaggle/data/train/train_label.npy 
    
    bucket: cyrusmv-sagemaker-demos source file: visa-kaggle/data/; ==> local: ../data//visa-kaggle/data/ 
    
    WARNING: cyrusmv-sagemaker-demos/visa-kaggle/data/ is a directory, skipping download operation
    bucket: cyrusmv-sagemaker-demos source file: visa-kaggle/data/recordio-pb-data; ==> local: ../data//visa-kaggle/data/recordio-pb-data 
    
    
    download completed.





    {'recordio-pb-data': '../data//visa-kaggle/data/recordio-pb-data',
     'train_data': '../data//visa-kaggle/data/train/train_data.npy',
     'train_label': '../data//visa-kaggle/data/train/train_label.npy',
     'val_data': '../data//visa-kaggle/data/test/val_data.npy',
     'val_label': '../data//visa-kaggle/data/test/val_label.npy'}



# Loading Data into Vectors
We will need to have the train and validation data to be loaded into numpy vectors before oriessing them.


```python
train_data = np.load(files['train_data'])
train_label = np.load(files['train_label'])

val_data = np.load(files['val_data'])
val_label = np.load(files['val_label'])

print("training data shape= {}; training label shape = {} \nValidation data shape= {}; validation label shape = {}".format(train_data.shape, 
                                                                        train_label.shape,
                                                                        val_data.shape,
                                                                        val_label.shape))
train_set = (train_data, train_label)
test_set = (val_data, val_label)

```

    training data shape= (199364, 30); training label shape = (199364,) 
    Validation data shape= (85443, 30); validation label shape = (85443,)


# Converting the Data
Amazon Algorithms support csv and recordio/protobuf. recordio is faster than CSV and specially in algorithms that deal with sparse matrices.
In the below snippet I am using sagemaker.amazon.core library in order to convert my numpy arrays into protobuf recordIO.


```python
vectors = np.array([t.tolist() for t in train_set[0]]).astype('float32')
labels = np.array([t.tolist() for t in train_set[1]]).astype('float32')

buf = io.BytesIO()
smac.write_numpy_to_dense_tensor(buf, vectors, labels)
buf.seek(0)
```




    0



# Upload Training Data to S3
Now that we've created our recordIO-wrapped protobuf, we'll need to upload it to S3, so that Amazon SageMaker training can use it.


```python
key = 'recordio-pb-data'
boto3.resource('s3').Bucket(bucket).Object(os.path.join(dist, key)).upload_fileobj(buf)
s3_train_data = 's3://{}/{}{}'.format(bucket, dist, key)
print('uploaded training data location: {}'.format(s3_train_data))
```

    uploaded training data location: s3://cyrusmv-sagemaker-demos/visa-kaggle/data/recordio-pb-data


Let's also setup an output S3 location for the model artifact to be uploaded to after training in complete.


```python
output_location = 's3://{}/{}output'.format(bucket, s3_4_output)
print('training artifacts will be uploaded to: {}'.format(output_location))
```

    training artifacts will be uploaded to: s3://cyrusmv-sagemaker-demos/visa-kaggle/output


# Training the Model with New Hyper Parameters


```python
containers = {'us-west-2': '174872318107.dkr.ecr.us-west-2.amazonaws.com/linear-learner:latest',
              'us-east-1': '382416733822.dkr.ecr.us-east-1.amazonaws.com/linear-learner:latest',
              'us-east-2': '404615174143.dkr.ecr.us-east-2.amazonaws.com/linear-learner:latest',
              'eu-west-1': '438346466558.dkr.ecr.eu-west-1.amazonaws.com/linear-learner:latest'}
```


```python
sess = sagemaker.Session()

linear = sagemaker.estimator.Estimator(containers[boto3.Session().region_name],
                                       role, #S3 role, so the notebook can read the data and upload the model
                                       train_instance_count=1, #number of instances for training
                                       train_instance_type='ml.m4.xlarge', # type of training instance
                                       output_path=output_location, #s3 location for uploading trained mdoel
                                       sagemaker_session=sess)

linear.set_hyperparameters(feature_dim=30, #dataset has 30 columns (features)
                           predictor_type='binary_classifier', # we predict a binary value. it could have been regressor
                           mini_batch_size=200,
                           #making recall the selection criteria and changin calibration samples that are used for threshold setting
                           binary_classifier_model_selection_criteria = 'precision_at_target_recall', 
                           target_recall = 0.9   
                          )

linear.fit({'train': s3_train_data})
```

    INFO:sagemaker:Creating training-job with name: linear-learner-2018-02-26-12-55-21-014


    ..................................................................
    [31mDocker entrypoint called with argument(s): train[0m
    [31m[02/26/2018 13:00:44 INFO 139651068401472] Reading default configuration from /opt/amazon/lib/python2.7/site-packages/algorithm/default-input.json: {u'epochs': u'10', u'init_bias': u'0.0', u'lr_scheduler_factor': u'0.99', u'num_calibration_samples': u'10000000', u'_num_kv_servers': u'auto', u'use_bias': u'true', u'num_point_for_scaler': u'10000', u'_log_level': u'info', u'bias_lr_mult': u'10', u'lr_scheduler_step': u'100', u'init_method': u'uniform', u'init_sigma': u'0.01', u'lr_scheduler_minimum_lr': u'0.00001', u'target_recall': u'0.8', u'num_models': u'32', u'early_stopping_patience': u'3', u'momentum': u'0.0', u'unbias_label': u'auto', u'wd': u'0.0', u'optimizer': u'adam', u'early_stopping_tolerance': u'0.001', u'learning_rate': u'auto', u'_kvstore': u'auto', u'normalize_data': u'true', u'binary_classifier_model_selection_criteria': u'accuracy', u'use_lr_scheduler': u'true', u'target_precision': u'0.8', u'force_dense': u'true', u'unbias_data': u'auto', u'init_scale': u'0.07', u'bias_wd_mult': u'0', u'mini_batch_size': u'1000', u'beta_1': u'0.9', u'loss': u'auto', u'beta_2': u'0.999', u'normalize_label': u'auto', u'_num_gpus': u'auto', u'_data_format': u'record', u'positive_example_weight_mult': u'1.0', u'l1': u'0.0'}[0m
    [31m[02/26/2018 13:00:44 INFO 139651068401472] Reading provided configuration from /opt/ml/input/config/hyperparameters.json: {u'target_recall': u'0.9', u'feature_dim': u'30', u'mini_batch_size': u'200', u'predictor_type': u'binary_classifier', u'binary_classifier_model_selection_criteria': u'precision_at_target_recall'}[0m
    [31m[02/26/2018 13:00:44 INFO 139651068401472] Final configuration: {u'epochs': u'10', u'feature_dim': u'30', u'init_bias': u'0.0', u'lr_scheduler_factor': u'0.99', u'num_calibration_samples': u'10000000', u'_num_kv_servers': u'auto', u'use_bias': u'true', u'num_point_for_scaler': u'10000', u'_log_level': u'info', u'bias_lr_mult': u'10', u'lr_scheduler_step': u'100', u'init_method': u'uniform', u'init_sigma': u'0.01', u'lr_scheduler_minimum_lr': u'0.00001', u'target_recall': u'0.9', u'num_models': u'32', u'early_stopping_patience': u'3', u'momentum': u'0.0', u'unbias_label': u'auto', u'wd': u'0.0', u'optimizer': u'adam', u'early_stopping_tolerance': u'0.001', u'learning_rate': u'auto', u'_kvstore': u'auto', u'normalize_data': u'true', u'binary_classifier_model_selection_criteria': u'precision_at_target_recall', u'use_lr_scheduler': u'true', u'target_precision': u'0.8', u'force_dense': u'true', u'unbias_data': u'auto', u'init_scale': u'0.07', u'bias_wd_mult': u'0', u'mini_batch_size': u'200', u'beta_1': u'0.9', u'loss': u'auto', u'beta_2': u'0.999', u'predictor_type': u'binary_classifier', u'normalize_label': u'auto', u'_num_gpus': u'auto', u'_data_format': u'record', u'positive_example_weight_mult': u'1.0', u'l1': u'0.0'}[0m
    [31m[02/26/2018 13:00:44 WARNING 139651068401472] Loggers have already been setup.[0m
    [31m[02/26/2018 13:00:44 INFO 139651068401472] Using default worker.[0m
    [31m[02/26/2018 13:00:44 INFO 139651068401472] Loaded iterator creator application/x-recordio-protobuf for content type ('application/x-recordio-protobuf', '1.0')[0m
    [31m[02/26/2018 13:00:44 INFO 139651068401472] Create Store: local[0m
    [31m[02/26/2018 13:00:44 WARNING 139651068401472] wait_for_all_workers will not sync workers since the kv store is not running distributed[0m
    [31m[02/26/2018 13:00:44 INFO 139651068401472] nvidia-smi took: 0.0251750946045 secs to identify 0 gpus[0m
    [31m[02/26/2018 13:00:44 INFO 139651068401472] Number of GPUs being used: 0[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 51, "sum": 51.0, "min": 51}, "Number of Batches Since Last Reset": {"count": 1, "max": 0, "sum": 0.0, "min": 0}, "Number of Records Since Last Reset": {"count": 1, "max": 0, "sum": 0.0, "min": 0}, "Total Batches Seen": {"count": 1, "max": 52, "sum": 52.0, "min": 52}, "Total Records Seen": {"count": 1, "max": 10400, "sum": 10400.0, "min": 10400}, "Max Records Seen Between Resets": {"count": 1, "max": 10200, "sum": 10200.0, "min": 10200}, "Reset Count": {"count": 1, "max": 2, "sum": 2.0, "min": 2}}, "EndTime": 1519650044.506193, "Dimensions": {"Host": "algo-1", "Meta": "init_train_data_iter", "Operation": "training", "Algorithm": "Linear Learner"}, "StartTime": 1519650044.506162}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.03263368721856889, "sum": 0.03263368721856889, "min": 0.03263368721856889}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650063.66795, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1519650063.667865}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.03097438685132198, "sum": 0.03097438685132198, "min": 0.03097438685132198}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650063.668031, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1519650063.668017}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.032773738992082545, "sum": 0.032773738992082545, "min": 0.032773738992082545}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650063.668069, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1519650063.668059}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.030927742073634062, "sum": 0.030927742073634062, "min": 0.030927742073634062}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650063.668104, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1519650063.668094}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.008452754719743886, "sum": 0.008452754719743886, "min": 0.008452754719743886}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650063.668136, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1519650063.668126}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.009712150798777183, "sum": 0.009712150798777183, "min": 0.009712150798777183}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650063.668168, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1519650063.668158}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.008114106326521072, "sum": 0.008114106326521072, "min": 0.008114106326521072}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650063.668208, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1519650063.668199}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.009226878187539669, "sum": 0.009226878187539669, "min": 0.009226878187539669}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650063.668237, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1519650063.668228}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.032869489538262645, "sum": 0.032869489538262645, "min": 0.032869489538262645}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650063.668266, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1519650063.668257}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.031155571191754924, "sum": 0.031155571191754924, "min": 0.031155571191754924}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650063.668296, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1519650063.668287}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.032897585283620766, "sum": 0.032897585283620766, "min": 0.032897585283620766}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650063.668325, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1519650063.668316}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.03116643935471893, "sum": 0.03116643935471893, "min": 0.03116643935471893}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650063.668355, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1519650063.668346}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.007926066940412839, "sum": 0.007926066940412839, "min": 0.007926066940412839}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650063.668384, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1519650063.668375}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.008717467358464891, "sum": 0.008717467358464891, "min": 0.008717467358464891}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650063.668413, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1519650063.668404}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.007969700393166924, "sum": 0.007969700393166924, "min": 0.007969700393166924}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650063.668441, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1519650063.668432}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.00866009896701991, "sum": 0.00866009896701991, "min": 0.00866009896701991}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650063.66847, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1519650063.668461}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.03715733311801071, "sum": 0.03715733311801071, "min": 0.03715733311801071}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650063.668499, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1519650063.66849}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.035650733820046764, "sum": 0.035650733820046764, "min": 0.035650733820046764}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650063.668527, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1519650063.668519}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.03709540682175313, "sum": 0.03709540682175313, "min": 0.03709540682175313}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650063.668557, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1519650063.668548}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.035670092640571804, "sum": 0.035670092640571804, "min": 0.035670092640571804}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650063.668589, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1519650063.66858}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.013384296401164468, "sum": 0.013384296401164468, "min": 0.013384296401164468}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650063.668619, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1519650063.66861}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.013402316727045265, "sum": 0.013402316727045265, "min": 0.013402316727045265}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650063.668648, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1519650063.668639}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.013345349924255491, "sum": 0.013345349924255491, "min": 0.013345349924255491}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650063.668677, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1519650063.668668}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.013416417274740057, "sum": 0.013416417274740057, "min": 0.013416417274740057}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650063.668707, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1519650063.668697}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.03799945135791618, "sum": 0.03799945135791618, "min": 0.03799945135791618}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650063.668735, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1519650063.668727}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.036508235096572396, "sum": 0.036508235096572396, "min": 0.036508235096572396}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650063.668764, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1519650063.668755}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0379524305845839, "sum": 0.0379524305845839, "min": 0.0379524305845839}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650063.668793, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1519650063.668784}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.036523705469855824, "sum": 0.036523705469855824, "min": 0.036523705469855824}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650063.668823, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1519650063.668814}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.014299934016826776, "sum": 0.014299934016826776, "min": 0.014299934016826776}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650063.668853, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1519650063.668844}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.014330657125903241, "sum": 0.014330657125903241, "min": 0.014330657125903241}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650063.668917, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1519650063.668905}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.014321290961579872, "sum": 0.014321290961579872, "min": 0.014321290961579872}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650063.668947, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1519650063.668938}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01432419491537667, "sum": 0.01432419491537667, "min": 0.01432419491537667}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650063.668976, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1519650063.668967}
    [0m
    [31m[02/26/2018 13:01:03 INFO 139651068401472] Epoch 1: Loss improved.  Updating best model[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 997, "sum": 997.0, "min": 997}, "Number of Batches Since Last Reset": {"count": 1, "max": 997, "sum": 997.0, "min": 997}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 1049, "sum": 1049.0, "min": 1049}, "Total Records Seen": {"count": 1, "max": 209764, "sum": 209764.0, "min": 209764}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 3, "sum": 3.0, "min": 3}}, "EndTime": 1519650063.669673, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1519650063.669645}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005667652155470896, "sum": 0.005667652155470896, "min": 0.005667652155470896}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650083.074053, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1519650083.073974}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004825513667789808, "sum": 0.004825513667789808, "min": 0.004825513667789808}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650083.074134, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1519650083.074119}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0056669846651365005, "sum": 0.0056669846651365005, "min": 0.0056669846651365005}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650083.074174, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1519650083.074163}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004824565563081618, "sum": 0.004824565563081618, "min": 0.004824565563081618}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650083.074209, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1519650083.074199}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004970770967791388, "sum": 0.004970770967791388, "min": 0.004970770967791388}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650083.074242, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1519650083.074232}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.008675491298588852, "sum": 0.008675491298588852, "min": 0.008675491298588852}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650083.074273, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1519650083.074263}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005216436268664508, "sum": 0.005216436268664508, "min": 0.005216436268664508}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650083.074304, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1519650083.074294}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.00767886707016424, "sum": 0.00767886707016424, "min": 0.00767886707016424}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650083.074334, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1519650083.074325}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.006160631613678721, "sum": 0.006160631613678721, "min": 0.006160631613678721}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650083.074365, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1519650083.074355}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005395665867665565, "sum": 0.005395665867665565, "min": 0.005395665867665565}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650083.074397, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1519650083.074386}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0061593689651314515, "sum": 0.0061593689651314515, "min": 0.0061593689651314515}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650083.07445, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1519650083.074431}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005394183763164474, "sum": 0.005394183763164474, "min": 0.005394183763164474}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650083.074514, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1519650083.074498}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005340294639344316, "sum": 0.005340294639344316, "min": 0.005340294639344316}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650083.074572, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1519650083.074554}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.007432443398671846, "sum": 0.007432443398671846, "min": 0.007432443398671846}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650083.07462, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1519650083.074607}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005339805002036284, "sum": 0.005339805002036284, "min": 0.005339805002036284}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650083.07467, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1519650083.074653}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.007413594041947261, "sum": 0.007413594041947261, "min": 0.007413594041947261}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650083.074731, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1519650083.074712}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012193860537914866, "sum": 0.012193860537914866, "min": 0.012193860537914866}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650083.074792, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1519650083.074773}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011884089119911912, "sum": 0.011884089119911912, "min": 0.011884089119911912}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650083.074848, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1519650083.074831}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0121924881020703, "sum": 0.0121924881020703, "min": 0.0121924881020703}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650083.074884, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1519650083.074874}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01188427789533234, "sum": 0.01188427789533234, "min": 0.01188427789533234}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650083.074916, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1519650083.074906}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011849327059275654, "sum": 0.011849327059275654, "min": 0.011849327059275654}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650083.074946, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1519650083.074936}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011949613026631285, "sum": 0.011949613026631285, "min": 0.011949613026631285}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650083.074976, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1519650083.074966}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011849455899861921, "sum": 0.011849455899861921, "min": 0.011849455899861921}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650083.075006, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1519650083.074996}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011950119458340916, "sum": 0.011950119458340916, "min": 0.011950119458340916}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650083.075046, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1519650083.075031}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.013074348031816233, "sum": 0.013074348031816233, "min": 0.013074348031816233}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650083.075104, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1519650083.075086}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012751769284347453, "sum": 0.012751769284347453, "min": 0.012751769284347453}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650083.075161, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1519650083.075142}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.013074507048151579, "sum": 0.013074507048151579, "min": 0.013074507048151579}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650083.07522, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1519650083.075202}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012752794072361117, "sum": 0.012752794072361117, "min": 0.012752794072361117}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650083.075275, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1519650083.075258}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012768974797837587, "sum": 0.012768974797837587, "min": 0.012768974797837587}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650083.075315, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1519650083.075303}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012907140801462482, "sum": 0.012907140801462482, "min": 0.012907140801462482}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650083.075346, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1519650083.075336}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012768395969412294, "sum": 0.012768395969412294, "min": 0.012768395969412294}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650083.075375, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1519650083.075366}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012902638967070414, "sum": 0.012902638967070414, "min": 0.012902638967070414}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650083.075405, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1519650083.075396}
    [0m
    [31m[02/26/2018 13:01:23 INFO 139651068401472] Epoch 2: Loss improved.  Updating best model[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 997, "sum": 997.0, "min": 997}, "Number of Batches Since Last Reset": {"count": 1, "max": 997, "sum": 997.0, "min": 997}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 2046, "sum": 2046.0, "min": 2046}, "Total Records Seen": {"count": 1, "max": 409128, "sum": 409128.0, "min": 409128}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 4, "sum": 4.0, "min": 4}}, "EndTime": 1519650083.076126, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1519650083.076095}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004935127349054239, "sum": 0.004935127349054239, "min": 0.004935127349054239}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650102.52742, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1519650102.527341}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.00427203314891272, "sum": 0.00427203314891272, "min": 0.00427203314891272}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650102.527499, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1519650102.527485}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004935854540382284, "sum": 0.004935854540382284, "min": 0.004935854540382284}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650102.527539, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1519650102.527528}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0042721039782012684, "sum": 0.0042721039782012684, "min": 0.0042721039782012684}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650102.527573, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1519650102.527563}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004420376588014445, "sum": 0.004420376588014445, "min": 0.004420376588014445}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650102.527606, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1519650102.527596}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.007742584361584883, "sum": 0.007742584361584883, "min": 0.007742584361584883}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650102.527638, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1519650102.527628}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.00456565413040569, "sum": 0.00456565413040569, "min": 0.00456565413040569}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650102.527668, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1519650102.527659}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.008615878523778665, "sum": 0.008615878523778665, "min": 0.008615878523778665}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650102.527699, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1519650102.527689}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005545444158784836, "sum": 0.005545444158784836, "min": 0.005545444158784836}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650102.52773, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1519650102.52772}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005075607791797344, "sum": 0.005075607791797344, "min": 0.005075607791797344}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650102.527761, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1519650102.527752}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005544712980648121, "sum": 0.005544712980648121, "min": 0.005544712980648121}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650102.527792, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1519650102.527782}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005074913500393011, "sum": 0.005074913500393011, "min": 0.005074913500393011}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650102.527822, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1519650102.527813}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005148838025304568, "sum": 0.005148838025304568, "min": 0.005148838025304568}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650102.527854, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1519650102.527844}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.007153353458508872, "sum": 0.007153353458508872, "min": 0.007153353458508872}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650102.527884, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1519650102.527874}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0051488738560143965, "sum": 0.0051488738560143965, "min": 0.0051488738560143965}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650102.527925, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1519650102.527905}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.007180837833740557, "sum": 0.007180837833740557, "min": 0.007180837833740557}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650102.527954, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1519650102.527945}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011892645105599878, "sum": 0.011892645105599878, "min": 0.011892645105599878}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650102.527983, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1519650102.527974}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011835506619758395, "sum": 0.011835506619758395, "min": 0.011835506619758395}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650102.528012, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1519650102.528003}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01189212998098518, "sum": 0.01189212998098518, "min": 0.01189212998098518}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650102.528042, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1519650102.528033}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01183551145306074, "sum": 0.01183551145306074, "min": 0.01183551145306074}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650102.528071, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1519650102.528062}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01188685285279071, "sum": 0.01188685285279071, "min": 0.01188685285279071}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650102.5281, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1519650102.528091}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012027238690253182, "sum": 0.012027238690253182, "min": 0.012027238690253182}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650102.528129, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1519650102.52812}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01188756551324244, "sum": 0.01188756551324244, "min": 0.01188756551324244}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650102.528158, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1519650102.528149}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012028014669344427, "sum": 0.012028014669344427, "min": 0.012028014669344427}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650102.528187, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1519650102.528178}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012809303724263088, "sum": 0.012809303724263088, "min": 0.012809303724263088}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650102.528217, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1519650102.528207}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012705335468053818, "sum": 0.012705335468053818, "min": 0.012705335468053818}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650102.528246, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1519650102.528237}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012809425910194237, "sum": 0.012809425910194237, "min": 0.012809425910194237}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650102.528275, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1519650102.528266}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012705433423440141, "sum": 0.012705433423440141, "min": 0.012705433423440141}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650102.528304, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1519650102.528295}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0127453378004482, "sum": 0.0127453378004482, "min": 0.0127453378004482}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650102.528334, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1519650102.528324}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.013007398396655618, "sum": 0.013007398396655618, "min": 0.013007398396655618}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650102.528363, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1519650102.528354}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012743747473570956, "sum": 0.012743747473570956, "min": 0.012743747473570956}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650102.528392, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1519650102.528383}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.013002959373537975, "sum": 0.013002959373537975, "min": 0.013002959373537975}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650102.528421, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1519650102.528412}
    [0m
    [31m[02/26/2018 13:01:42 INFO 139651068401472] Epoch 3: Loss improved.  Updating best model[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 997, "sum": 997.0, "min": 997}, "Number of Batches Since Last Reset": {"count": 1, "max": 997, "sum": 997.0, "min": 997}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 3043, "sum": 3043.0, "min": 3043}, "Total Records Seen": {"count": 1, "max": 608492, "sum": 608492.0, "min": 608492}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 5, "sum": 5.0, "min": 5}}, "EndTime": 1519650102.529137, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1519650102.529107}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.00470667933693612, "sum": 0.00470667933693612, "min": 0.00470667933693612}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650121.662357, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1519650121.662279}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004219565484186552, "sum": 0.004219565484186552, "min": 0.004219565484186552}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650121.662438, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1519650121.662424}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004706364989609843, "sum": 0.004706364989609843, "min": 0.004706364989609843}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650121.662487, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1519650121.662469}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0042195765129159615, "sum": 0.0042195765129159615, "min": 0.0042195765129159615}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650121.662528, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1519650121.662517}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0041763336680470465, "sum": 0.0041763336680470465, "min": 0.0041763336680470465}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650121.662563, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1519650121.662552}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.007466715085052513, "sum": 0.007466715085052513, "min": 0.007466715085052513}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650121.662595, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1519650121.662585}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004113265507645067, "sum": 0.004113265507645067, "min": 0.004113265507645067}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650121.662626, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1519650121.662616}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.007283831990726505, "sum": 0.007283831990726505, "min": 0.007283831990726505}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650121.662673, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1519650121.662661}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005365034565030332, "sum": 0.005365034565030332, "min": 0.005365034565030332}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650121.662718, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1519650121.662708}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005026118026290792, "sum": 0.005026118026290792, "min": 0.005026118026290792}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650121.662748, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1519650121.662739}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005364483495403725, "sum": 0.005364483495403725, "min": 0.005364483495403725}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650121.662778, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1519650121.662769}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005025832318517578, "sum": 0.005025832318517578, "min": 0.005025832318517578}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650121.662812, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1519650121.662799}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005029672592145253, "sum": 0.005029672592145253, "min": 0.005029672592145253}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650121.662854, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1519650121.662843}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.007021656825107122, "sum": 0.007021656825107122, "min": 0.007021656825107122}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650121.662884, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1519650121.662875}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005029573528372858, "sum": 0.005029573528372858, "min": 0.005029573528372858}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650121.662914, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1519650121.662905}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.007024514108662957, "sum": 0.007024514108662957, "min": 0.007024514108662957}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650121.662944, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1519650121.662935}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01182302532575456, "sum": 0.01182302532575456, "min": 0.01182302532575456}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650121.662974, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1519650121.662964}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011836684630757354, "sum": 0.011836684630757354, "min": 0.011836684630757354}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650121.663017, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1519650121.663006}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011822717206245925, "sum": 0.011822717206245925, "min": 0.011822717206245925}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650121.663048, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1519650121.663039}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011836674486298158, "sum": 0.011836674486298158, "min": 0.011836674486298158}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650121.663078, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1519650121.663069}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011847777807838227, "sum": 0.011847777807838227, "min": 0.011847777807838227}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650121.663107, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1519650121.663098}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012101647110943903, "sum": 0.012101647110943903, "min": 0.012101647110943903}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650121.663139, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1519650121.663128}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011848036126199975, "sum": 0.011848036126199975, "min": 0.011848036126199975}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650121.663184, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1519650121.663173}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012102348985128015, "sum": 0.012102348985128015, "min": 0.012102348985128015}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650121.663216, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1519650121.663206}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01275710071156542, "sum": 0.01275710071156542, "min": 0.01275710071156542}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650121.663246, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1519650121.663236}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012708294382655477, "sum": 0.012708294382655477, "min": 0.012708294382655477}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650121.663275, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1519650121.663266}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012756788470000627, "sum": 0.012756788470000627, "min": 0.012756788470000627}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650121.663305, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1519650121.663296}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012708063989309182, "sum": 0.012708063989309182, "min": 0.012708063989309182}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650121.663349, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1519650121.663338}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012723198192844908, "sum": 0.012723198192844908, "min": 0.012723198192844908}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650121.66338, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1519650121.66337}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.013092594677563681, "sum": 0.013092594677563681, "min": 0.013092594677563681}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650121.66341, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1519650121.663401}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012723153346184506, "sum": 0.012723153346184506, "min": 0.012723153346184506}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650121.66344, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1519650121.663431}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.013088389750398007, "sum": 0.013088389750398007, "min": 0.013088389750398007}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650121.663469, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1519650121.66346}
    [0m
    [31m[02/26/2018 13:02:01 INFO 139651068401472] Epoch 4: Loss improved.  Updating best model[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 997, "sum": 997.0, "min": 997}, "Number of Batches Since Last Reset": {"count": 1, "max": 997, "sum": 997.0, "min": 997}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 4040, "sum": 4040.0, "min": 4040}, "Total Records Seen": {"count": 1, "max": 807856, "sum": 807856.0, "min": 807856}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 6, "sum": 6.0, "min": 6}}, "EndTime": 1519650121.664203, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1519650121.664166}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004611710490262413, "sum": 0.004611710490262413, "min": 0.004611710490262413}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650140.604067, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1519650140.603988}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004105323628114769, "sum": 0.004105323628114769, "min": 0.004105323628114769}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650140.604149, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1519650140.604135}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.00459792397916317, "sum": 0.00459792397916317, "min": 0.00459792397916317}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650140.604188, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1519650140.604178}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004105155112737992, "sum": 0.004105155112737992, "min": 0.004105155112737992}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650140.604222, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1519650140.604212}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0039766494365573585, "sum": 0.0039766494365573585, "min": 0.0039766494365573585}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650140.604255, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1519650140.604245}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0075506342755196154, "sum": 0.0075506342755196154, "min": 0.0075506342755196154}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650140.604287, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1519650140.604277}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.003935648076999259, "sum": 0.003935648076999259, "min": 0.003935648076999259}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650140.604318, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1519650140.604309}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.007110261109482305, "sum": 0.007110261109482305, "min": 0.007110261109482305}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650140.604349, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1519650140.60434}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005293915902365403, "sum": 0.005293915902365403, "min": 0.005293915902365403}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650140.60438, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1519650140.604371}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005014482273993243, "sum": 0.005014482273993243, "min": 0.005014482273993243}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650140.604421, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1519650140.604411}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0052860826690752346, "sum": 0.0052860826690752346, "min": 0.0052860826690752346}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650140.60445, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1519650140.604441}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005014375631050412, "sum": 0.005014375631050412, "min": 0.005014375631050412}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650140.604481, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1519650140.604471}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004961481665646336, "sum": 0.004961481665646336, "min": 0.004961481665646336}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650140.60451, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1519650140.604501}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.006869401255113832, "sum": 0.006869401255113832, "min": 0.006869401255113832}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650140.604539, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1519650140.60453}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004961449588652715, "sum": 0.004961449588652715, "min": 0.004961449588652715}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650140.604569, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1519650140.604559}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.006865836331643254, "sum": 0.006865836331643254, "min": 0.006865836331643254}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650140.604598, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1519650140.604589}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01180566990007478, "sum": 0.01180566990007478, "min": 0.01180566990007478}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650140.604627, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1519650140.604618}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01183925409797564, "sum": 0.01183925409797564, "min": 0.01183925409797564}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650140.604658, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1519650140.604648}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011803128483095561, "sum": 0.011803128483095561, "min": 0.011803128483095561}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650140.604687, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1519650140.604677}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01183924244410063, "sum": 0.01183924244410063, "min": 0.01183924244410063}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650140.604716, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1519650140.604707}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011829299801384589, "sum": 0.011829299801384589, "min": 0.011829299801384589}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650140.604745, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1519650140.604736}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012156352428625896, "sum": 0.012156352428625896, "min": 0.012156352428625896}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650140.604775, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1519650140.604765}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011829381115644811, "sum": 0.011829381115644811, "min": 0.011829381115644811}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650140.604804, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1519650140.604795}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012156713170478173, "sum": 0.012156713170478173, "min": 0.012156713170478173}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650140.604834, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1519650140.604825}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012739559324480683, "sum": 0.012739559324480683, "min": 0.012739559324480683}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650140.604898, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1519650140.604854}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01271223040454718, "sum": 0.01271223040454718, "min": 0.01271223040454718}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650140.604935, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1519650140.604924}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012737001866610892, "sum": 0.012737001866610892, "min": 0.012737001866610892}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650140.604964, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1519650140.604955}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012712316866440466, "sum": 0.012712316866440466, "min": 0.012712316866440466}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650140.604995, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1519650140.604986}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012713815574066705, "sum": 0.012713815574066705, "min": 0.012713815574066705}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650140.605025, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1519650140.605015}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.013131026259920818, "sum": 0.013131026259920818, "min": 0.013131026259920818}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650140.605054, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1519650140.605045}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012713467052213878, "sum": 0.012713467052213878, "min": 0.012713467052213878}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650140.605083, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1519650140.605074}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.013129017615586101, "sum": 0.013129017615586101, "min": 0.013129017615586101}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650140.605112, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1519650140.605103}
    [0m
    [31m[02/26/2018 13:02:20 INFO 139651068401472] Epoch 5: Loss improved.  Updating best model[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 997, "sum": 997.0, "min": 997}, "Number of Batches Since Last Reset": {"count": 1, "max": 997, "sum": 997.0, "min": 997}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 5037, "sum": 5037.0, "min": 5037}, "Total Records Seen": {"count": 1, "max": 1007220, "sum": 1007220.0, "min": 1007220}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 7, "sum": 7.0, "min": 7}}, "EndTime": 1519650140.605797, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1519650140.605767}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004567881404874794, "sum": 0.004567881404874794, "min": 0.004567881404874794}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650159.643498, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1519650159.643419}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004108989967788797, "sum": 0.004108989967788797, "min": 0.004108989967788797}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650159.643579, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1519650159.643564}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004496083612737047, "sum": 0.004496083612737047, "min": 0.004496083612737047}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650159.643618, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1519650159.643607}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004109522250581936, "sum": 0.004109522250581936, "min": 0.004109522250581936}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650159.643653, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1519650159.643643}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.003910600187268153, "sum": 0.003910600187268153, "min": 0.003910600187268153}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650159.643685, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1519650159.643675}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.006552148453970057, "sum": 0.006552148453970057, "min": 0.006552148453970057}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650159.643716, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1519650159.643706}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0038893299709900795, "sum": 0.0038893299709900795, "min": 0.0038893299709900795}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650159.643747, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1519650159.643737}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.007003359099528397, "sum": 0.007003359099528397, "min": 0.007003359099528397}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650159.64378, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1519650159.64377}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005262415757081116, "sum": 0.005262415757081116, "min": 0.005262415757081116}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650159.643811, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1519650159.643802}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005009079959812054, "sum": 0.005009079959812054, "min": 0.005009079959812054}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650159.643842, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1519650159.643833}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005213723903499454, "sum": 0.005213723903499454, "min": 0.005213723903499454}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650159.643873, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1519650159.643863}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005009045689983899, "sum": 0.005009045689983899, "min": 0.005009045689983899}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650159.643904, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1519650159.643894}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004925963813775634, "sum": 0.004925963813775634, "min": 0.004925963813775634}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650159.643934, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1519650159.643924}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.006781070482674949, "sum": 0.006781070482674949, "min": 0.006781070482674949}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650159.643965, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1519650159.643955}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004925954038703657, "sum": 0.004925954038703657, "min": 0.004925954038703657}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650159.644005, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1519650159.643996}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.006782465332513114, "sum": 0.006782465332513114, "min": 0.006782465332513114}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650159.644035, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1519650159.644026}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011795597286767748, "sum": 0.011795597286767748, "min": 0.011795597286767748}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650159.644064, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1519650159.644055}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01184286235401549, "sum": 0.01184286235401549, "min": 0.01184286235401549}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650159.644094, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1519650159.644085}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011787788577407718, "sum": 0.011787788577407718, "min": 0.011787788577407718}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650159.644124, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1519650159.644114}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011842852172153782, "sum": 0.011842852172153782, "min": 0.011842852172153782}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650159.644153, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1519650159.644144}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011788364233202245, "sum": 0.011788364233202245, "min": 0.011788364233202245}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650159.644183, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1519650159.644174}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012187641067841804, "sum": 0.012187641067841804, "min": 0.012187641067841804}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650159.644212, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1519650159.644203}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011788384169190523, "sum": 0.011788384169190523, "min": 0.011788384169190523}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650159.644244, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1519650159.644232}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01218779618234699, "sum": 0.01218779618234699, "min": 0.01218779618234699}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650159.644294, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1519650159.644277}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012732436974274826, "sum": 0.012732436974274826, "min": 0.012732436974274826}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650159.644344, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1519650159.644328}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012717095222548548, "sum": 0.012717095222548548, "min": 0.012717095222548548}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650159.644386, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1519650159.644371}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01272172955639032, "sum": 0.01272172955639032, "min": 0.01272172955639032}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650159.644432, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1519650159.644416}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012716884806931736, "sum": 0.012716884806931736, "min": 0.012716884806931736}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650159.644477, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1519650159.644462}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012706778104525493, "sum": 0.012706778104525493, "min": 0.012706778104525493}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650159.644526, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1519650159.64451}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.013123839060162535, "sum": 0.013123839060162535, "min": 0.013123839060162535}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650159.644567, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1519650159.644552}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01270658638926754, "sum": 0.01270658638926754, "min": 0.01270658638926754}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650159.644611, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1519650159.644596}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.013123013289172172, "sum": 0.013123013289172172, "min": 0.013123013289172172}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650159.644651, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1519650159.644636}
    [0m
    [31m[02/26/2018 13:02:39 INFO 139651068401472] Epoch 6: Loss improved.  Updating best model[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 997, "sum": 997.0, "min": 997}, "Number of Batches Since Last Reset": {"count": 1, "max": 997, "sum": 997.0, "min": 997}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 6034, "sum": 6034.0, "min": 6034}, "Total Records Seen": {"count": 1, "max": 1206584, "sum": 1206584.0, "min": 1206584}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 8, "sum": 8.0, "min": 8}}, "EndTime": 1519650159.645366, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1519650159.645337}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004545867700937642, "sum": 0.004545867700937642, "min": 0.004545867700937642}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650179.086655, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1519650179.086576}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004049344359650221, "sum": 0.004049344359650221, "min": 0.004049344359650221}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650179.086736, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1519650179.086722}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0043974114005195806, "sum": 0.0043974114005195806, "min": 0.0043974114005195806}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650179.086795, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1519650179.086778}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004049361949902491, "sum": 0.004049361949902491, "min": 0.004049361949902491}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650179.086839, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1519650179.086824}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0038826522287579786, "sum": 0.0038826522287579786, "min": 0.0038826522287579786}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650179.086879, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1519650179.086865}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.006586752429345752, "sum": 0.006586752429345752, "min": 0.006586752429345752}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650179.086919, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1519650179.086904}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.003856992545784507, "sum": 0.003856992545784507, "min": 0.003856992545784507}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650179.086957, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1519650179.086943}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0075348123487346454, "sum": 0.0075348123487346454, "min": 0.0075348123487346454}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650179.086993, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1519650179.08698}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0052472872146401535, "sum": 0.0052472872146401535, "min": 0.0052472872146401535}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650179.08704, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1519650179.087017}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005004265089650231, "sum": 0.005004265089650231, "min": 0.005004265089650231}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650179.087078, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1519650179.087063}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0051453708563314144, "sum": 0.0051453708563314144, "min": 0.0051453708563314144}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650179.087113, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1519650179.087101}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005004256397306201, "sum": 0.005004256397306201, "min": 0.005004256397306201}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650179.087149, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1519650179.087136}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004907966210585402, "sum": 0.004907966210585402, "min": 0.004907966210585402}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650179.087184, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1519650179.087171}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0066095224192438936, "sum": 0.0066095224192438936, "min": 0.0066095224192438936}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650179.087219, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1519650179.087207}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004907936070443516, "sum": 0.004907936070443516, "min": 0.004907936070443516}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650179.087255, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1519650179.087242}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.006611413776345765, "sum": 0.006611413776345765, "min": 0.006611413776345765}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650179.087294, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1519650179.087281}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011789075090104796, "sum": 0.011789075090104796, "min": 0.011789075090104796}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650179.087331, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1519650179.087317}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01184605859384599, "sum": 0.01184605859384599, "min": 0.01184605859384599}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650179.087368, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1519650179.087353}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011777278435786805, "sum": 0.011777278435786805, "min": 0.011777278435786805}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650179.087404, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1519650179.087391}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011846051086862402, "sum": 0.011846051086862402, "min": 0.011846051086862402}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650179.087441, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1519650179.087426}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011769279434858077, "sum": 0.011769279434858077, "min": 0.011769279434858077}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650179.087479, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1519650179.087464}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012179446628870794, "sum": 0.012179446628870794, "min": 0.012179446628870794}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650179.087518, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1519650179.087502}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011769299665278938, "sum": 0.011769299665278938, "min": 0.011769299665278938}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650179.087555, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1519650179.087542}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012179503373925137, "sum": 0.012179503373925137, "min": 0.012179503373925137}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650179.087591, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1519650179.087578}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012728928114216491, "sum": 0.012728928114216491, "min": 0.012728928114216491}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650179.087626, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1519650179.087613}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012720759077424027, "sum": 0.012720759077424027, "min": 0.012720759077424027}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650179.087663, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1519650179.087648}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012711438687002084, "sum": 0.012711438687002084, "min": 0.012711438687002084}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650179.087699, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1519650179.087686}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01272067763794019, "sum": 0.01272067763794019, "min": 0.01272067763794019}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650179.087734, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1519650179.087721}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012704518941571435, "sum": 0.012704518941571435, "min": 0.012704518941571435}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650179.087768, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1519650179.087756}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.013100485013027387, "sum": 0.013100485013027387, "min": 0.013100485013027387}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650179.087806, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1519650179.08779}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012704512371445995, "sum": 0.012704512371445995, "min": 0.012704512371445995}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650179.087843, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1519650179.08783}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.013100207350590832, "sum": 0.013100207350590832, "min": 0.013100207350590832}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650179.087878, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1519650179.087865}
    [0m
    [31m[02/26/2018 13:02:59 INFO 139651068401472] Epoch 7: Loss improved.  Updating best model[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 997, "sum": 997.0, "min": 997}, "Number of Batches Since Last Reset": {"count": 1, "max": 997, "sum": 997.0, "min": 997}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 7031, "sum": 7031.0, "min": 7031}, "Total Records Seen": {"count": 1, "max": 1405948, "sum": 1405948.0, "min": 1405948}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 9, "sum": 9.0, "min": 9}}, "EndTime": 1519650179.088569, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1519650179.088538}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004526676941722871, "sum": 0.004526676941722871, "min": 0.004526676941722871}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650199.028148, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1519650199.028068}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004051024555509169, "sum": 0.004051024555509169, "min": 0.004051024555509169}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650199.028227, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1519650199.028214}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004315813462133987, "sum": 0.004315813462133987, "min": 0.004315813462133987}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650199.028287, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1519650199.028268}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004050180803550535, "sum": 0.004050180803550535, "min": 0.004050180803550535}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650199.028341, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1519650199.028323}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0038712834243486865, "sum": 0.0038712834243486865, "min": 0.0038712834243486865}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650199.028386, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1519650199.02837}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.006589103323905334, "sum": 0.006589103323905334, "min": 0.006589103323905334}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650199.028429, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1519650199.028414}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0038505053970528894, "sum": 0.0038505053970528894, "min": 0.0038505053970528894}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650199.028484, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1519650199.028459}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005915358228500031, "sum": 0.005915358228500031, "min": 0.005915358228500031}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650199.028525, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1519650199.02851}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005234768807319513, "sum": 0.005234768807319513, "min": 0.005234768807319513}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650199.028565, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1519650199.02855}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004998800254064092, "sum": 0.004998800254064092, "min": 0.004998800254064092}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650199.028605, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1519650199.028591}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005089589439063187, "sum": 0.005089589439063187, "min": 0.005089589439063187}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650199.028652, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1519650199.028636}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.00499879976065014, "sum": 0.00499879976065014, "min": 0.00499879976065014}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650199.028692, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1519650199.028677}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.00490084696172112, "sum": 0.00490084696172112, "min": 0.00490084696172112}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650199.028732, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1519650199.028717}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.006530738547701486, "sum": 0.006530738547701486, "min": 0.006530738547701486}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650199.028771, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1519650199.028757}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004905023862828452, "sum": 0.004905023862828452, "min": 0.004905023862828452}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650199.028816, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1519650199.0288}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.006530863705145412, "sum": 0.006530863705145412, "min": 0.006530863705145412}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650199.028857, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1519650199.028842}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011786108918368338, "sum": 0.011786108918368338, "min": 0.011786108918368338}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650199.028944, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1519650199.028926}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011847347200036528, "sum": 0.011847347200036528, "min": 0.011847347200036528}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650199.028982, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1519650199.028969}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011772062216986375, "sum": 0.011772062216986375, "min": 0.011772062216986375}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650199.02902, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1519650199.029006}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011847343250779982, "sum": 0.011847343250779982, "min": 0.011847343250779982}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650199.029056, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1519650199.029043}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01176816618616083, "sum": 0.01176816618616083, "min": 0.01176816618616083}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650199.029095, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1519650199.029079}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012139096920299961, "sum": 0.012139096920299961, "min": 0.012139096920299961}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650199.029131, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1519650199.029118}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011768902741432429, "sum": 0.011768902741432429, "min": 0.011768902741432429}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650199.029166, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1519650199.029154}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012139115441893509, "sum": 0.012139115441893509, "min": 0.012139115441893509}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650199.029202, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1519650199.029189}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012726223468780518, "sum": 0.012726223468780518, "min": 0.012726223468780518}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650199.029241, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1519650199.029225}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01272210651253122, "sum": 0.01272210651253122, "min": 0.01272210651253122}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650199.029276, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1519650199.029263}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012706393007054386, "sum": 0.012706393007054386, "min": 0.012706393007054386}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650199.029314, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1519650199.029298}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012722163508425515, "sum": 0.012722163508425515, "min": 0.012722163508425515}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650199.029352, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1519650199.029336}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012702869241047337, "sum": 0.012702869241047337, "min": 0.012702869241047337}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650199.02939, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1519650199.029377}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.013068538757272992, "sum": 0.013068538757272992, "min": 0.013068538757272992}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650199.029427, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1519650199.029413}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012703835455678313, "sum": 0.012703835455678313, "min": 0.012703835455678313}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650199.029463, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1519650199.02945}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.013068433111144537, "sum": 0.013068433111144537, "min": 0.013068433111144537}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650199.029499, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1519650199.029485}
    [0m
    [31m[02/26/2018 13:03:19 INFO 139651068401472] Epoch 8: Loss improved.  Updating best model[0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 997, "sum": 997.0, "min": 997}, "Number of Batches Since Last Reset": {"count": 1, "max": 997, "sum": 997.0, "min": 997}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 8028, "sum": 8028.0, "min": 8028}, "Total Records Seen": {"count": 1, "max": 1605312, "sum": 1605312.0, "min": 1605312}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 10, "sum": 10.0, "min": 10}}, "EndTime": 1519650199.030182, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1519650199.030152}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004505495541470956, "sum": 0.004505495541470956, "min": 0.004505495541470956}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650219.623476, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1519650219.623397}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004010293081058675, "sum": 0.004010293081058675, "min": 0.004010293081058675}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650219.623554, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1519650219.62354}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004255344651893799, "sum": 0.004255344651893799, "min": 0.004255344651893799}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650219.623616, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1519650219.623598}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004009794415569449, "sum": 0.004009794415569449, "min": 0.004009794415569449}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650219.623668, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1519650219.623651}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0038669205634447048, "sum": 0.0038669205634447048, "min": 0.0038669205634447048}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650219.62371, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1519650219.623696}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.006008855093728254, "sum": 0.006008855093728254, "min": 0.006008855093728254}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650219.623754, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1519650219.623735}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.00384946655515716, "sum": 0.00384946655515716, "min": 0.00384946655515716}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650219.623795, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1519650219.623781}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005872559198451857, "sum": 0.005872559198451857, "min": 0.005872559198451857}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650219.623832, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1519650219.623819}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005221268843754707, "sum": 0.005221268843754707, "min": 0.005221268843754707}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650219.62387, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1519650219.623856}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004992413520588573, "sum": 0.004992413520588573, "min": 0.004992413520588573}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650219.623908, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1519650219.623893}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005048058870506574, "sum": 0.005048058870506574, "min": 0.005048058870506574}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650219.623946, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1519650219.623932}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004992413541758395, "sum": 0.004992413541758395, "min": 0.004992413541758395}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650219.623993, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1519650219.623969}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004898119423179382, "sum": 0.004898119423179382, "min": 0.004898119423179382}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650219.624028, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1519650219.624015}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.006300828982641112, "sum": 0.006300828982641112, "min": 0.006300828982641112}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650219.624063, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1519650219.62405}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004904547084841024, "sum": 0.004904547084841024, "min": 0.004904547084841024}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650219.624101, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1519650219.624086}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.006300837922056793, "sum": 0.006300837922056793, "min": 0.006300837922056793}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650219.624138, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1519650219.624125}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011783242104163611, "sum": 0.011783242104163611, "min": 0.011783242104163611}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650219.624174, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1519650219.62416}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01184659194261075, "sum": 0.01184659194261075, "min": 0.01184659194261075}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650219.624212, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1519650219.624198}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011770077425253439, "sum": 0.011770077425253439, "min": 0.011770077425253439}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650219.624247, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1519650219.624234}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01184659024737925, "sum": 0.01184659024737925, "min": 0.01184659024737925}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650219.624283, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1519650219.62427}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011764718556260488, "sum": 0.011764718556260488, "min": 0.011764718556260488}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650219.624319, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1519650219.624305}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012089582515682233, "sum": 0.012089582515682233, "min": 0.012089582515682233}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650219.624358, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1519650219.624342}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011769188922721458, "sum": 0.011769188922721458, "min": 0.011769188922721458}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650219.624394, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1519650219.624381}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012089588386679898, "sum": 0.012089588386679898, "min": 0.012089588386679898}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650219.62443, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1519650219.624417}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012723508089870574, "sum": 0.012723508089870574, "min": 0.012723508089870574}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650219.624465, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1519650219.624452}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012721733154811295, "sum": 0.012721733154811295, "min": 0.012721733154811295}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650219.624502, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1519650219.624487}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012704492840900958, "sum": 0.012704492840900958, "min": 0.012704492840900958}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650219.624537, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1519650219.624525}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01272176590340444, "sum": 0.01272176590340444, "min": 0.01272176590340444}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650219.624572, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1519650219.62456}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01270238969893939, "sum": 0.01270238969893939, "min": 0.01270238969893939}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650219.624607, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1519650219.624594}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.013035495221525072, "sum": 0.013035495221525072, "min": 0.013035495221525072}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650219.624642, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1519650219.624629}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01270381375503947, "sum": 0.01270381375503947, "min": 0.01270381375503947}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650219.624677, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1519650219.624665}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.013035462547849161, "sum": 0.013035462547849161, "min": 0.013035462547849161}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650219.624712, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1519650219.624699}
    [0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 997, "sum": 997.0, "min": 997}, "Number of Batches Since Last Reset": {"count": 1, "max": 997, "sum": 997.0, "min": 997}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 9025, "sum": 9025.0, "min": 9025}, "Total Records Seen": {"count": 1, "max": 1804676, "sum": 1804676.0, "min": 1804676}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 11, "sum": 11.0, "min": 11}}, "EndTime": 1519650219.624958, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1519650219.624928}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004484348957871936, "sum": 0.004484348957871936, "min": 0.004484348957871936}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650239.849232, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1519650239.849153}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004022592038222884, "sum": 0.004022592038222884, "min": 0.004022592038222884}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650239.849303, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1519650239.849289}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004211591286846552, "sum": 0.004211591286846552, "min": 0.004211591286846552}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650239.849343, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1519650239.849332}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004024088239357115, "sum": 0.004024088239357115, "min": 0.004024088239357115}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650239.849378, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1519650239.849368}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0038653407705925317, "sum": 0.0038653407705925317, "min": 0.0038653407705925317}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650239.849411, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1519650239.849401}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005845328790589655, "sum": 0.005845328790589655, "min": 0.005845328790589655}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650239.849444, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1519650239.849434}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.00384838671634819, "sum": 0.00384838671634819, "min": 0.00384838671634819}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650239.849475, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1519650239.849466}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.006996697691346595, "sum": 0.006996697691346595, "min": 0.006996697691346595}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650239.849506, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1519650239.849497}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005207857994295387, "sum": 0.005207857994295387, "min": 0.005207857994295387}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650239.849538, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1519650239.849528}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004985605237339754, "sum": 0.004985605237339754, "min": 0.004985605237339754}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650239.849568, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1519650239.849559}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005017690771973756, "sum": 0.005017690771973756, "min": 0.005017690771973756}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650239.849599, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1519650239.849589}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004985605736064863, "sum": 0.004985605736064863, "min": 0.004985605736064863}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650239.849637, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1519650239.849622}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004897197788379278, "sum": 0.004897197788379278, "min": 0.004897197788379278}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650239.849698, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1519650239.849686}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.00601727135074568, "sum": 0.00601727135074568, "min": 0.00601727135074568}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650239.849748, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1519650239.849732}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0049041122156962096, "sum": 0.0049041122156962096, "min": 0.0049041122156962096}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650239.849803, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1519650239.849785}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0060172773374386705, "sum": 0.0060172773374386705, "min": 0.0060172773374386705}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650239.849856, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1519650239.849838}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01178060655268919, "sum": 0.01178060655268919, "min": 0.01178060655268919}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650239.849906, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1519650239.849888}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011844644571344059, "sum": 0.011844644571344059, "min": 0.011844644571344059}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650239.849966, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1519650239.849947}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011769445296601358, "sum": 0.011769445296601358, "min": 0.011769445296601358}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650239.850028, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1519650239.850008}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011844643962431145, "sum": 0.011844643962431145, "min": 0.011844643962431145}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650239.850088, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1519650239.850069}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011761395003960793, "sum": 0.011761395003960793, "min": 0.011761395003960793}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650239.85015, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1519650239.85013}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012047038233416327, "sum": 0.012047038233416327, "min": 0.012047038233416327}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650239.850187, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1519650239.850177}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011769212809611995, "sum": 0.011769212809611995, "min": 0.011769212809611995}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650239.850219, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1519650239.850209}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012047040044745229, "sum": 0.012047040044745229, "min": 0.012047040044745229}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650239.85025, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1519650239.850241}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012720975050156614, "sum": 0.012720975050156614, "min": 0.012720975050156614}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650239.850281, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1519650239.850271}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012720586984571684, "sum": 0.012720586984571684, "min": 0.012720586984571684}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650239.850311, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1519650239.850301}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012703913864810543, "sum": 0.012703913864810543, "min": 0.012703913864810543}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650239.850347, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1519650239.850332}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012720593931864064, "sum": 0.012720593931864064, "min": 0.012720593931864064}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650239.850406, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1519650239.850387}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01270207726362958, "sum": 0.01270207726362958, "min": 0.01270207726362958}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650239.850466, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1519650239.850447}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.013001564804008268, "sum": 0.013001564804008268, "min": 0.013001564804008268}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650239.850529, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1519650239.850509}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012703781923705556, "sum": 0.012703781923705556, "min": 0.012703781923705556}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650239.85059, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1519650239.850572}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.013001554854902397, "sum": 0.013001554854902397, "min": 0.013001554854902397}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1519650239.850628, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1519650239.850617}
    [0m
    [31m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 997, "sum": 997.0, "min": 997}, "Number of Batches Since Last Reset": {"count": 1, "max": 997, "sum": 997.0, "min": 997}, "Number of Records Since Last Reset": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Total Batches Seen": {"count": 1, "max": 10022, "sum": 10022.0, "min": 10022}, "Total Records Seen": {"count": 1, "max": 2004040, "sum": 2004040.0, "min": 2004040}, "Max Records Seen Between Resets": {"count": 1, "max": 199364, "sum": 199364.0, "min": 199364}, "Reset Count": {"count": 1, "max": 12, "sum": 12.0, "min": 12}}, "EndTime": 1519650239.85082, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1519650239.850773}
    [0m
    [31m[02/26/2018 13:03:59 WARNING 139651068401472] wait_for_all_workers will not sync workers since the kv store is not running distributed[0m
    [31m[02/26/2018 13:03:59 WARNING 139651068401472] wait_for_all_workers will not sync workers since the kv store is not running distributed[0m
    [31m#metrics {"Metrics": {"f1": {"count": 1, "max": 0.7965116279069767, "sum": 0.7965116279069767, "min": 0.7965116279069767}, "threshold_for_accuracy": {"count": 1, "max": 0.09889889508485794, "sum": 0.09889889508485794, "min": 0.09889889508485794}, "threshold_for_precision_at_target_recall": {"count": 1, "max": 0.0027813639026135206, "sum": 0.0027813639026135206, "min": 0.0027813639026135206}, "threshold_for_recall_at_target_precision": {"count": 1, "max": 0.09889889508485794, "sum": 0.09889889508485794, "min": 0.09889889508485794}, "recall_at_precision": {"count": 1, "max": 0.7826086956522369, "sum": 0.7826086956522369, "min": 0.7826086956522369}, "precision_at_target_recall": {"count": 1, "max": 0.07998971193418004, "sum": 0.07998971193418004, "min": 0.07998971193418004}, "accuracy": {"count": 1, "max": 0.9993027828494613, "sum": 0.9993027828494613, "min": 0.9993027828494613}, "threshold_for_f1": {"count": 1, "max": 0.0868278294801712, "sum": 0.0868278294801712, "min": 0.0868278294801712}}, "EndTime": 1519650241.707622, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner"}, "StartTime": 1519650241.691832}
    [0m
    [31m[02/26/2018 13:04:01 INFO 139651068401472] Selection criteria: precision_at_target_recall[0m
    [31mmodel: 0[0m
    [31mthreshold: 0.002781[0m
    [31mscore: 0.079990[0m
    [31m[02/26/2018 13:04:01 INFO 139651068401472] Saved checkpoint to "/tmp/tmpP7lQZu/mx-mod-0000.params"[0m
    [31m[02/26/2018 13:04:01 INFO 139651068401472] Test data was not provided.[0m
    [31m#metrics {"Metrics": {"totaltime": {"count": 1, "max": 197432.461977005, "sum": 197432.461977005, "min": 197432.461977005}, "finalize.time": {"count": 1, "max": 1862.950086593628, "sum": 1862.950086593628, "min": 1862.950086593628}, "initialize.time": {"count": 1, "max": 149.4441032409668, "sum": 149.4441032409668, "min": 149.4441032409668}, "check_early_stopping.time": {"count": 10, "max": 0.6079673767089844, "sum": 4.834651947021484, "min": 0.0629425048828125}, "setuptime": {"count": 1, "max": 21.54684066772461, "sum": 21.54684066772461, "min": 21.54684066772461}, "update.time": {"count": 10, "max": 20594.554901123047, "sum": 195342.83208847046, "min": 18941.43581390381}, "epochs": {"count": 1, "max": 10, "sum": 10.0, "min": 10}}, "EndTime": 1519650241.720329, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner"}, "StartTime": 1519650044.345995}
    [0m
    ===== Job Complete =====


# Hosting


```python
linear_predictor = linear.deploy(initial_instance_count=1, #Initial number of instances. 
                                                           #Autoscaling can increase the number of instances.
                                 instance_type='ml.m4.xlarge') # instance type
```

    INFO:sagemaker:Creating model with name: linear-learner-2018-02-26-13-10-07-983
    INFO:sagemaker:Creating endpoint with name linear-learner-2018-02-26-12-55-21-014


    ----------------------------------------------------------------------------------------------------------------------------!


```python
type(linear_predictor)
```




    sagemaker.predictor.RealTimePredictor



# Prediction


```python
linear_predictor.content_type = 'text/csv'
linear_predictor.serializer = csv_serializer
linear_predictor.deserializer = json_deserializer
```


```python
predictions = []
for array in np.array_split(test_set[0], 100):
    result = linear_predictor.predict(array)
    predictions += [r['predicted_label'] for r in result['predictions']]

predictions = np.array(predictions)
```


```python
import pandas as pd

pd.crosstab(test_set[1], predictions, rownames=['actuals'], colnames=['predictions'])
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>predictions</th>
      <th>0.0</th>
      <th>1.0</th>
    </tr>
    <tr>
      <th>actuals</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0</th>
      <td>83778</td>
      <td>1518</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>20</td>
      <td>127</td>
    </tr>
  </tbody>
</table>
</div>




```python
print("false positive after Hyper-Parameter change = {}".format(127/(20+127)))
print("false positive before Hyper-Parameter change = {}".format(85443/(85443+1518)))

```

    false positive after Hyper-Parameter change = 0.8639455782312925
    false positive before Hyper-Parameter change = 0.982543898989202


# Analyzing the Results
The confusion matrix above indicates that:
- Total fraudulent transactions: 147
 - Num Examples (NE) = 85443
 - True Positive (TP) = 127
 - False Positive (FP) = 1518
 - False Negative (FN) = 20

- **Recall** = TP/(TP+FN) = 127/(127+20) = 0.86
- **Precision** = TP/(TP+FP) = 127/(127+1518) = 0.08
- **Accuracy** = 1- (FP+FN)/NE = 1 - (1538/85443) = 0.98

recall on fraud is this mode is 86% as opposed to 80% with default parameters. This is a significant improvement on recall, even though precision has now dropped to a very low value.
An important fact to notice is that from parallel models, in this model, model #0 and in the model with default values model #12, yielded the best results. This is testament to the power of HPO, which SageMaker provides out of the box.

Using HPO have significantly shorten experiment time, thus releasing your scientists to work on new problems while reducing time to market for your model.

| After Changing Hyper-Parameters | Before Changing Hyper-Parameters|
|:--------------------------------|:--------------------------------|
| model: 0                        | model: 12                       |
| threshold: 0.002781             | threshold: 0.028                |
| score: 0.079990                 | score: 0.999418                 |


# Conclusions
By optimizing hyperparameters we have significantly improved recall, but precision is lost. It could work well for this example give our goal and distribution of data. Other examples might be different. This dataset needs to be enriched to further improve the resi


# (optional) Delete the endpoint
f you're ready to be done with this notebook, please run the delete_endpoint line in the cell below. This will remove the hosted endpoint you created and avoid any charges from a stray instance being left on.


```python
linear.delete_endpoint()
```

# End of Part 3. 
PREVIOUS: [Part2: Getting Hands-On with Linear Learner and Amazon SageMaker](linearlearner-blogpost-part2.ipynb)
