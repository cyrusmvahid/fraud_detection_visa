
# Getting Hands-On with Linear Learner and Amazon SageMaker

# Assumptions and Disclaimers
This blogpost assumes that you have already completed the following tutorials from Amazon SageMaker docuemntation:
- [Setting up](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-set-up.html)
- [Create am Amazon SageMaker Notebook Instance](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-setup-working-env.html)
- I have included 'sagemaker' in the name of my S3 bucket, "cyrusmv-sagemaker-demos' and have chosen to let any SageMaker notebook instance to access any S3 bucket with the term 'sagemaker' included in the name. This is however is not a recommended security option for production and is only useful for simplifying the flow of the blog.
- I have downloaded [Visa dataset from Kaggle](https://www.kaggle.com/dalpozz/creditcardfraud), have renamed it to original.csv, and have uploaded it to a S3 bucket. You would need to do the same.
- It is assumed that the reader is familiar with linear regression. If not please read part 1 of this post.



# Introduction

In this part of the blog, I will provide an introduction to inner-workings of SageMaker and Amazon LinearLearner algorithm. Then I will download [Visa dataset from Kaggle](https://www.kaggle.com/dalpozz/creditcardfraud) from an S3 location to my notebook instanceand pre-process it in order to feed the data to the algorithm.
We then create a live endpoint and make predictions using trained models.

Amazon SageMaker
Amazon SageMaker is a fully managed machine learning service that automates the end-to-end ML process. With Amazon SageMaker, data scientists and developers can quickly and easily build and train machine learning models and directly deploy them into a production-ready hosted environment. It provides an integrated Jupyter authoring notebook instance for easy access to your data sources for exploration and analysis, so you don't have to manage servers. It also provides common machine learning algorithms that are optimized to run efficiently against extremely large data in a distributed environment. With native support for bring-your-own-algorithms and frameworks, Amazon SageMaker offers flexible distributed training options that adjust to your specific workflows. Deploy a model into a secure and scalable environment by launching it with a single click from the Amazon SageMaker console. Training and hosting are billed by minutes of usage, with no minimum fees and no upfront commitments.
SageMaker python sdk, is a library that provides the user with the ability to wrap your models in helper classes that can be passed to Amazon SageMaker. The python sdk provides you with the ability to use Amazon Algorithms, train and deploy your own code (currently MXNet and TensorFlow) using Amazon SagedMaker, and bring your own pre-trained dockerized models.
Let us next take a look at the architecture of SageMaker.
How does Amazon SageMaker Work:
Training
Training workloads, for an end-to-end model development includes the following components:
•	S3 Bucket for the training dataset
•	Amazon SageMaker notebook instance, a fully managed single tenant EC2 instance.
•	EC2 cluster for training. (fully managed transparent to developer)
•	S3 bucket for trained model
Hosting
•	S3 bucket for model
•	ECS infrastructure to host a dockerized model (fully managed transparent to developer)
•	EC2 cluster manages by ECS
•	Model Endpoints, elastically scalable callable inference endpoints. (fully managed transparent to developer)


![SageMaker Architecture](https://docs.aws.amazon.com/sagemaker/latest/dg/images/ironman-architecture.png)

## The SageMaker Process - Training
1. Upload the data into S3. Prior to this we go through several preparation steps that are necessary, but are not specific to Amazon SageMaker. 
2. Train: Training code is wrapped inside sagemaker helper code. In case of Amazon Algorithms this is already completed and instead of using your own training and wrapper code, you can simply pull the model from [Amazon ECR (Elastic Container Registry)](https://aws.amazon.com/ecr/). During the training process, the following steps will happen:
 1. A training cluster based on requested specification will be created. 
 Dataset will be streamed to the training cluster
    - Authentication and Authorization will be based on execution role of the SageMaker 
 2. A training job is launched.
 3. Data is streamed to the training instances.
 4- Hyperparameters are passed to the training model.
 5- Training will end after a termination criteria is reached.
2. After the model is trained, model artefacts will be uploaded to S3.
3. The training cluster is torn down.


### Get the data
1. Download the dataset from [kaggle](https://www.kaggle.com/dalpozz/creditcardfraud) to your local machine.
2. Upload the dataset onto S3. *reminder*: I am using an S3 bucket whose name includes the term *"sagemaker"*
The rest of this blog assumes that you have already done this.
### Inspecting and understanding the data 
*NOTE:* If you are interested in learning about preprocessing data, you should start here, otherwise you could simply start from part 2, the SageMaker Pipeline, when we load the data from npy files.
1. Download the dataset from S3 onto your notebook instance
2. Loading the csv file into a pandas dataframe for inspection


```python
#imports
import boto3 #AWS python SDK for accessing AWS services
import pandas as pd #Tabular data structure
import numpy as np #Array libraru with probability and statistics capabilities
import matplotlib.pyplot as plt # Plotting library
import seaborn as sns #plotting library
import io
import sagemaker.amazon.common as smac # Amazon Sagemaker common library that includes data formats
import sagemaker #sagemaker python sdk
import os
from sagemaker.predictor import csv_serializer, json_deserializer #sagemaker prediction sdk
```


```python
bucket = 'cyrusmv-sagemaker-demos'     #replace this with your own bucket 
original_key = 'visa-kaggle/original.csv'    #replace this with your own file inside the bucket
protocol="s3://"
local_origin_data = '/tmp/original.csv'
local_pickel_root = '../data/'
dist = 'visa-kaggle/data/'
s3_4_output = 'visa-kaggle/'
 
# Define IAM role
#import boto3
#import re
from sagemaker import get_execution_role
role = get_execution_role() #this is SageMaker role that would be later used for authorizing SageMaker to access S3
print(role) 

sagemaker_session = sagemaker.Session()
```

    arn:aws:iam::475933981307:role/service-role/AmazonSageMaker-ExecutionRole-20180102T172706


### Pre-Processing
pre-processing included several steps
1. Downloding file locally
2. Loading file into pandas for inspection
3. conversing pandas DataFrame to numpy
4. shuffling the data
5. spliting data into test and training
6. breaking up each data set to data and label


```python
#Downloading the file to a local folder
s3 = boto3.resource('s3')
s3.Bucket(bucket).download_file(original_key, local_origin_data)
os.listdir('/tmp')
```




    ['hsperfdata_role-agent',
     'original.csv',
     '.ICE-unix',
     'titanic_train.csv',
     'visa-kaggle']




```python
# loading data into pandas for inspection
df = pd.read_csv(local_origin_data)
print(df.as_matrix().shape)
```

    (284807, 31)


### Data Dimension
Shape of data is $284807 \times 31$, meaning that $dim(input\_data)\ = \ 1$. Next we would like to distinguish features from the target.


```python
df.head(5)
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
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>-1.359807</td>
      <td>-0.072781</td>
      <td>2.536347</td>
      <td>1.378155</td>
      <td>-0.338321</td>
      <td>0.462388</td>
      <td>0.239599</td>
      <td>0.098698</td>
      <td>0.363787</td>
      <td>...</td>
      <td>-0.018307</td>
      <td>0.277838</td>
      <td>-0.110474</td>
      <td>0.066928</td>
      <td>0.128539</td>
      <td>-0.189115</td>
      <td>0.133558</td>
      <td>-0.021053</td>
      <td>149.62</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1.191857</td>
      <td>0.266151</td>
      <td>0.166480</td>
      <td>0.448154</td>
      <td>0.060018</td>
      <td>-0.082361</td>
      <td>-0.078803</td>
      <td>0.085102</td>
      <td>-0.255425</td>
      <td>...</td>
      <td>-0.225775</td>
      <td>-0.638672</td>
      <td>0.101288</td>
      <td>-0.339846</td>
      <td>0.167170</td>
      <td>0.125895</td>
      <td>-0.008983</td>
      <td>0.014724</td>
      <td>2.69</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>-1.358354</td>
      <td>-1.340163</td>
      <td>1.773209</td>
      <td>0.379780</td>
      <td>-0.503198</td>
      <td>1.800499</td>
      <td>0.791461</td>
      <td>0.247676</td>
      <td>-1.514654</td>
      <td>...</td>
      <td>0.247998</td>
      <td>0.771679</td>
      <td>0.909412</td>
      <td>-0.689281</td>
      <td>-0.327642</td>
      <td>-0.139097</td>
      <td>-0.055353</td>
      <td>-0.059752</td>
      <td>378.66</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>-0.966272</td>
      <td>-0.185226</td>
      <td>1.792993</td>
      <td>-0.863291</td>
      <td>-0.010309</td>
      <td>1.247203</td>
      <td>0.237609</td>
      <td>0.377436</td>
      <td>-1.387024</td>
      <td>...</td>
      <td>-0.108300</td>
      <td>0.005274</td>
      <td>-0.190321</td>
      <td>-1.175575</td>
      <td>0.647376</td>
      <td>-0.221929</td>
      <td>0.062723</td>
      <td>0.061458</td>
      <td>123.50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>-1.158233</td>
      <td>0.877737</td>
      <td>1.548718</td>
      <td>0.403034</td>
      <td>-0.407193</td>
      <td>0.095921</td>
      <td>0.592941</td>
      <td>-0.270533</td>
      <td>0.817739</td>
      <td>...</td>
      <td>-0.009431</td>
      <td>0.798278</td>
      <td>-0.137458</td>
      <td>0.141267</td>
      <td>-0.206010</td>
      <td>0.502292</td>
      <td>0.219422</td>
      <td>0.215153</td>
      <td>69.99</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>




```python
# We would like to see what values we have for "Class". This snippet below shows that Class is a binary column.
print()
print(df.groupby('Class').size())
```

    
    Class
    0    284315
    1       492
    dtype: int64


### Feature and Target identification
*"Class"* is our target and the rest of the columns are our features. Next we would like to understand values or clusters of *"Class"*.

This tells us that we are predicting a binary target and this instead of linear regression, would should be performing a logistic regression or binary classification.

So far we know that:
- Of the 31 input dimentions, 30 are features and *'Class'* is the target.
- The target is a binary prediction
thus our lineare regression function would be defined as:

$$
f:\mathbb{R}^{30} \rightarrow \{0, 1\} \\
f(x) = W_{30 \times 1} V_{1 \times 30} + b
$$


```python
num_recs = df['Class'].count()
num_zeros = df['Class'][df['Class']==0].count()
num_ones = num_recs - num_zeros
print("{}% of transactions are fraudunat and {}% are legitimate".format((num_ones/num_recs)*100,
                                                                       (num_zeros/num_recs*100)))

```

    0.1727485630620034% of transactions are fraudunat and 99.82725143693798% are legitimate


### Data Normalization
As the last observation we would like to check data distribution as whether or not the data is notmalized by plotting all the columns. We often would like the data to be within a comparable range with the same center and perhaps similar standard deviation, so numerical value ranges would not create bias towards some features. This however is not always the case. For instance if we would like one featire to have a bigger impact, we can simply let it. Below os the box plot from our data set. We can see that apart from 'Time' and 'Amount', the rest of the data is normalized, so I assume Visa wanted 'Time' and 'Amount' to have a much higher impact.

***AmazonSageMaker Tip: Amazon LinearLearner normalized the data by default. If your would like to prevent data normalization you will have to change the ser the value of*** `normalize_data` ***to*** `false` ***in the hyperparameters.***


```python
df.boxplot(figsize=(20, 10))
plt.show()
```


![png](output_17_0.png)


Dropping "Time", "Amount", and "Class", we can observer the distribution with greater detail. Here not all columns are centered around the same point (0 in this case)


```python
df1 = df.drop(['Time', 'Amount', 'Class'], axis=1)
df1.boxplot(figsize=(20, 10))
plt.show()

```


![png](output_19_0.png)


### Data Preparation
Before attempting to prepare the data, we would like to inspect distribution of fraud/not-fraud values for class. We know from last two code snippets that **99.8** percent of our transactions are legitimate, so it is crucial to randomize distribution of Class==1 within the dataset, so we do not end up with a lumpy concentration of most of Class==1 in a segment of dataset 

To randomize distribution of Class==1, we shuffle the data in the dataset.

There are three issues with Class==1 being concentrated near one another:
1. We might learn sequential noise
2. We might unlearn what we have learned as we iterate down in the dataset.
3. The dataset includes both training and validation sets and we need to split the data. with a distribution that is not normal and smooth, we might end up with most or all of Class==1 in training or validation.



### Splitting the data to data and label
Next, we would need to prepare the data through the following steps.
1.	Convert tabular data to a numpy vector, so that we can manipulate the data easier.
2.	Split the data to data and label, but moving the "Class" or indeed the last column of the numpy array to a new array.


```python
#Converting Data Into Numpy

raw_data = df.as_matrix()

#Shuffling the data and randomizing the distrbution of the data
#I have performed to shuffles and experimented with different seeds until the distribution 
#of Class==1 became acceptably smooth according to the graph below.

np.random.seed(123)
np.random.shuffle(raw_data)
np.random.seed(499)
np.random.shuffle(raw_data)


label = raw_data[:, -1] #Taking last column of the data and creating a lanel vector
data = raw_data[:, :-1] # Taking the remains of th da

print("shape before split: {}; label_shape = {}; data_shape= {}".format(raw_data.shape, label.shape, data.shape))
```

    shape before split: (284807, 31); label_shape = (284807,); data_shape= (284807, 30)


From previous snippet we see that the originl dataset includes 31 dimensional rows and now we have two vectors, the data vector with each row having a dimension of 30 and label is just a scalar.

### Plotting distribution of Class==1 
Next we would like to be sure that Class==1 records are distributed evenly within the dataset before we attempt to split the data to training and validation.


```python
# There are very few fraudulant transactions in the dataset, so I am putting their indexes in an array
# to plot and ensure they are evenly distributed, so when I split the dataset into test and training 
# I don't end up with a dispropostionate distribution
t = []
for i in range(len(label)):
    if label[i] == 1:
        t.append(i)

sns.distplot(t, kde=True, rug=True, hist=False)
plt.show()
```


![png](output_24_0.png)


The plot above shows is the distribution of Class==1. the blue bar in the bottom (rug==True) shows us that Class==1 is evenly distributed within the range of the data [0..284807]
From the plot above we can observe that 

### Splitting the dataset to train and validation
Now that we are in posession of a good dataset, we cam proceed to split the data into training and validation sets. Here I have done a 70-30 split, you can choose whatever rate you find appropriage.


```python
#Splitting data into validation and training and breaking dataset into data and label

#70%-30% training to validation
train_size = int(data.shape[0]*0.7)

#training data and associated labels
train_data  = data[:train_size, :]
val_data = data[train_size:, :]

#validation data and associated labels
train_label = label[:train_size]
val_label = label[train_size:]


print("training data shape= {}; training label shape = {} \nValidation data shape= {}; validation label shape = {}".format(train_data.shape, 
                                                                        train_label.shape,
                                                                        val_data.shape,
                                                                        val_label.shape))
```

    training data shape= (199364, 30); training label shape = (199364,) 
    Validation data shape= (85443, 30); validation label shape = (85443,)


### Saving data for later use
Here I strongly recommend to save the resulting data into files and uploading them to S3. This is a toy dataset and is quickly processed. In real-life pre-processing of the data could take a long time. I remember once I did pre-process a language corpus into vectors and did not save them. I had to later restart my notebooks server and lost all the value in notebook memory. This was 6 hours of wait time.
The saving and uploading are performed in two separate steps:

1. Saving the vectors, suing numpy.save, which saves the prepared vectors as npy objects. I have set `pickle=True` to use pickle serialization.
2. Uploading the data from local (notebook server) to S3. I am using sgemaker.Session object, so that the upload would be authorized to my S3 bucket. Output of sagemaker.upload_data indicates where the files are located. You can check your S3 bucket to ensure the upload is perfomed fully using


```python
#Saving arrays for later use
np.save(local_pickel_root + 'train/train_data.npy', train_data, allow_pickle=True)
np.save(local_pickel_root + 'train/train_label.npy', train_label, allow_pickle=True)
np.save(local_pickel_root + 'test/val_data.npy', val_data, allow_pickle=True)
np.save(local_pickel_root + 'test/val_label.npy', val_label, allow_pickle=True)

!ls -R '../data/'

```

    ../data/:
    test  train
    
    ../data/test:
    val_data.npy  val_label.npy
    
    ../data/train:
    train_data.npy	train_label.npy


# Part2 - SageMaker Pipeline
## Upload your data onto S3
This could have been done outside of sagemaker as well. 


```python
#Uploading the data.
'''
path is the local path on your notbooks instance
bucket is your bucket name
key_prefix is your folder structure inside you S3 bucket
'''
S3loc = sagemaker_session.upload_data(path=local_pickel_root, bucket=bucket, key_prefix='visa-kaggle/data')
print(S3loc)
!aws s3 ls cyrusmv-sagemaker-demos/visa-kaggle/data/ --recursive #use the output from your own S3loc
```

    s3://cyrusmv-sagemaker-demos/visa-kaggle/data
    2018-01-04 12:19:10          0 visa-kaggle/data/
    2018-02-14 15:01:45   20506400 visa-kaggle/data/test/val_data.npy
    2018-02-14 15:01:45     683624 visa-kaggle/data/test/val_label.npy
    2018-02-14 15:01:44   47847440 visa-kaggle/data/train/train_data.npy
    2018-02-14 15:01:44    1594992 visa-kaggle/data/train/train_label.npy


### Loading Data into Vectors
We will need to have the train and validation data to be loaded into numpy vectors before using them. I am loading the vector from the files I have locally recorded and not S3


```python
train_data = np.load(local_pickel_root + 'train/train_data.npy')
train_label = np.load(local_pickel_root + 'train/train_label.npy')

val_data = np.load(local_pickel_root + 'test/val_data.npy')
val_label = np.load(local_pickel_root + 'test/val_label.npy')

print("training data shape= {}; training label shape = {} \nValidation data shape= {}; validation label shape = {}".format(train_data.shape, 
                                                                        train_label.shape,
                                                                        val_data.shape,
                                                                        val_label.shape))
train_set = (train_data, train_label)
test_set = (val_data, val_label)

```

    training data shape= (199364, 30); training label shape = (199364,) 
    Validation data shape= (85443, 30); validation label shape = (85443,)


### Data Conversion
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



### Upload training data
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


### End of Data Preparation Phase
At this point we have a set of npy training and validation files in S3 and no longer need the local data. This tutorial up to here can be split into a separate notebook. In fact I strongly recommend that you should do this. Next, we are going to train LinearLearnier model with the Visa dataset; but before that, let me get a little bit deeper into SageMaker and Amazon Algorithms.

## SageMaker Proces - Training
Before diving into the code, let us get a little bit under the hood of SageMaker and see how it works. SageMaker python SDK has a few core Classes. The most important class hierarchy for training jobs is the Estimator class.

The base class, EstimatorBase, has a method, called ```python fit()```. All classes that are derived from EstimatorBase implement ```python fit()```.

Calling ```python fit()``` results in creating a training job, spinning up a training cluster, and training the model. Once the training is complete, the trained model artefacts will be saved in S3 and training cluster will be torn down.

### Estimator Class Hierarchy
Ignoring Frameworks and other built-in algorithm we can see that EstimatorBase branches into two sections, general Estimator, used for calling dockerized models, and FrameWork from which Estimators class for using frameworks such as MXNet and TensorFlow are derived.
![EstimatorHierarchy](../images/estimator.png)

Training a model in SageMaker consists of creating an Estimator object, setting hyperparameters, and calling ```python fit()```. Later we review some of the more important hyperparameters.

```python
linear = sagemaker.estimator.Estimator(containers[boto3.Session().region_name],
                                       role, #S3 role, so the notebook can read the data and upload the model
                                       train_instance_count=1, #number of instances for training
                                       train_instance_type='ml.p2.xlarge', # type of training instance
                                       output_path=output_location, #s3 location for uploading trained mdoel
                                       sagemaker_session=sess)
linear.set_hyperparameters(feature_dim=30, #dataset has 30 columns (features)
                           predictor_type='binary_classifier', # for regression set to regressor
                           mini_batch_size=200)

linear.fit({'train': s3_train_data})                                
                                    
```

SageMaker python SDK code can be found in [github](https://github.com/aws/sagemaker-python-sdk). Exploring the code, you can find another alternative to calling Estimator and passing your model image to it. in sagemaker-python-sdk/src/sagemaker/amazon/ you will find a different class hierarchy in which a new class, AmazonAlgorithmsEstimatorBase is derived directly from EstimatorBase. Each of the algorithms implements AmazonAlgorithmsEstimatorBase. For instance, LinearLearner is a specific estimator based on this new superclass. We do not need to use these estimators since general estimator gives us a uniformed way of calling all dockerized, pre-trained algorithms. Below is the class hierarchy for LinearLearner class.

![estimator class](../images/amazonalgoestimator.png)

## SageMaker Process - Training the model
multiple regions. We follow the following steps
1. define containers dictionary.
2. Create am Estimator object and pass the hyper-parameters as well as the model location to it.
3. run Estimator.fit to begin training the model


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
                                       train_instance_type='ml.p2.xlarge', # type of training instance
                                       output_path=output_location, #s3 location for uploading trained mdoel
                                       sagemaker_session=sess)

linear.set_hyperparameters(feature_dim=30, #dataset has 30 columns (features)
                           predictor_type='binary_classifier', # we predict a binary value. it could have been regressor
                           mini_batch_size=200)

linear.fit({'train': s3_train_data})
```

    INFO:sagemaker:Creating training-job with name: linear-learner-2018-02-13-16-23-45-273


    ............................................................................
    [31mDocker entrypoint called with argument(s): train[0m
    [31m[02/13/2018 16:29:57 INFO 140471961589568] Reading default configuration from /opt/amazon/lib/python2.7/site-packages/algorithm/default-input.json: {u'epochs': u'10', u'init_bias': u'0.0', u'lr_scheduler_factor': u'0.99', u'num_calibration_samples': u'10000000', u'_num_kv_servers': u'auto', u'use_bias': u'true', u'num_point_for_scaler': u'10000', u'_log_level': u'info', u'bias_lr_mult': u'10', u'lr_scheduler_step': u'100', u'init_method': u'uniform', u'init_sigma': u'0.01', u'lr_scheduler_minimum_lr': u'0.00001', u'target_recall': u'0.8', u'num_models': u'32', u'momentum': u'0.0', u'unbias_label': u'auto', u'wd': u'0.0', u'optimizer': u'adam', u'learning_rate': u'auto', u'_kvstore': u'auto', u'normalize_data': u'true', u'binary_classifier_model_selection_criteria': u'accuracy', u'use_lr_scheduler': u'true', u'target_precision': u'0.8', u'force_dense': u'true', u'unbias_data': u'auto', u'init_scale': u'0.07', u'bias_wd_mult': u'0', u'mini_batch_size': u'1000', u'beta_1': u'0.9', u'loss': u'auto', u'beta_2': u'0.999', u'normalize_label': u'auto', u'_num_gpus': u'auto', u'_data_format': u'record', u'positive_example_weight_mult': u'1.0', u'l1': u'0.0'}[0m
    [31m[02/13/2018 16:29:57 INFO 140471961589568] Reading provided configuration from /opt/ml/input/config/hyperparameters.json: {u'feature_dim': u'30', u'mini_batch_size': u'200', u'predictor_type': u'binary_classifier'}[0m
    [31m[02/13/2018 16:29:57 INFO 140471961589568] Final configuration: {u'epochs': u'10', u'feature_dim': u'30', u'init_bias': u'0.0', u'lr_scheduler_factor': u'0.99', u'num_calibration_samples': u'10000000', u'_num_kv_servers': u'auto', u'use_bias': u'true', u'num_point_for_scaler': u'10000', u'_log_level': u'info', u'bias_lr_mult': u'10', u'lr_scheduler_step': u'100', u'init_method': u'uniform', u'init_sigma': u'0.01', u'lr_scheduler_minimum_lr': u'0.00001', u'target_recall': u'0.8', u'num_models': u'32', u'momentum': u'0.0', u'unbias_label': u'auto', u'wd': u'0.0', u'optimizer': u'adam', u'learning_rate': u'auto', u'_kvstore': u'auto', u'normalize_data': u'true', u'binary_classifier_model_selection_criteria': u'accuracy', u'use_lr_scheduler': u'true', u'target_precision': u'0.8', u'force_dense': u'true', u'unbias_data': u'auto', u'init_scale': u'0.07', u'bias_wd_mult': u'0', u'mini_batch_size': u'200', u'beta_1': u'0.9', u'loss': u'auto', u'beta_2': u'0.999', u'predictor_type': u'binary_classifier', u'normalize_label': u'auto', u'_num_gpus': u'auto', u'_data_format': u'record', u'positive_example_weight_mult': u'1.0', u'l1': u'0.0'}[0m
    [31m[02/13/2018 16:29:57 WARNING 140471961589568] Loggers have already been setup.[0m
    [31m[02/13/2018 16:29:57 INFO 140471961589568] Detected entry point for worker worker[0m
    [31m[02/13/2018 16:29:58 INFO 140471961589568] Loaded iterator creator application/x-recordio-protobuf for content type ('application/x-recordio-protobuf', '1.0')[0m
    [31m[02/13/2018 16:29:58 INFO 140471961589568] Create Store: local[0m
    [31m[02/13/2018 16:29:59 WARNING 140471961589568] wait_for_all_workers will not sync workers since the kv store is not running distributed[0m
    [31m[02/13/2018 16:29:59 INFO 140471961589568] nvidia-smi took: 0.0503001213074 secs to identify 1 gpus[0m
    [31m[02/13/2018 16:29:59 INFO 140471961589568] Number of GPUs being used: 1[0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.03263369270267496, "sum": 0.03263369270267496, "min": 0.03263369270267496}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539441.557544, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1518539441.557371}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.030974385090561277, "sum": 0.030974385090561277, "min": 0.030974385090561277}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539441.557693, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1518539441.557669}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.032773735335912095, "sum": 0.032773735335912095, "min": 0.032773735335912095}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539441.557762, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1518539441.557744}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.030927742364326393, "sum": 0.030927742364326393, "min": 0.030927742364326393}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539441.557827, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1518539441.557809}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.008452769120735299, "sum": 0.008452769120735299, "min": 0.008452769120735299}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539441.55789, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1518539441.557872}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.009712099996540343, "sum": 0.009712099996540343, "min": 0.009712099996540343}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539441.557946, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1518539441.557928}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.008114031247633798, "sum": 0.008114031247633798, "min": 0.008114031247633798}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539441.557999, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1518539441.557983}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.009226878358265777, "sum": 0.009226878358265777, "min": 0.009226878358265777}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539441.558053, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1518539441.558036}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.03286949151461144, "sum": 0.03286949151461144, "min": 0.03286949151461144}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539441.558106, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1518539441.55809}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.031155570906598165, "sum": 0.031155570906598165, "min": 0.031155570906598165}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539441.558166, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1518539441.558148}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.03289758820759963, "sum": 0.03289758820759963, "min": 0.03289758820759963}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539441.558228, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1518539441.55821}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.031166436637201943, "sum": 0.031166436637201943, "min": 0.031166436637201943}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539441.558289, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1518539441.558272}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.007926067693563866, "sum": 0.007926067693563866, "min": 0.007926067693563866}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539441.558348, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1518539441.558331}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.00871746794931239, "sum": 0.00871746794931239, "min": 0.00871746794931239}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539441.558402, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1518539441.558386}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.007969699952738307, "sum": 0.007969699952738307, "min": 0.007969699952738307}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539441.558457, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1518539441.55844}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.00866009907789966, "sum": 0.00866009907789966, "min": 0.00866009907789966}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539441.558512, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1518539441.558495}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.03715733054322172, "sum": 0.03715733054322172, "min": 0.03715733054322172}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539441.558567, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1518539441.55855}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.035650734449007425, "sum": 0.035650734449007425, "min": 0.035650734449007425}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539441.558623, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1518539441.558606}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.03709540521075209, "sum": 0.03709540521075209, "min": 0.03709540521075209}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539441.558679, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1518539441.558662}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.035670090340466865, "sum": 0.035670090340466865, "min": 0.035670090340466865}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539441.558733, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1518539441.558717}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.013384296694895754, "sum": 0.013384296694895754, "min": 0.013384296694895754}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539441.558789, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1518539441.558772}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.013402316943358025, "sum": 0.013402316943358025, "min": 0.013402316943358025}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539441.55885, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1518539441.558833}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.013345350097326271, "sum": 0.013345350097326271, "min": 0.013345350097326271}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539441.558908, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1518539441.558891}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01341641724495363, "sum": 0.01341641724495363, "min": 0.01341641724495363}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539441.558967, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1518539441.55895}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.03799945816427589, "sum": 0.03799945816427589, "min": 0.03799945816427589}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539441.559028, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1518539441.55901}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.03650823699213175, "sum": 0.03650823699213175, "min": 0.03650823699213175}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539441.559089, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1518539441.559071}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.03795243291042177, "sum": 0.03795243291042177, "min": 0.03795243291042177}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539441.55915, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1518539441.559133}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.036523704892061803, "sum": 0.036523704892061803, "min": 0.036523704892061803}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539441.559212, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1518539441.559194}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.014299934009266792, "sum": 0.014299934009266792, "min": 0.014299934009266792}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539441.559276, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1518539441.559258}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.014330657201816317, "sum": 0.014330657201816317, "min": 0.014330657201816317}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539441.559334, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1518539441.559317}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.014321291175807441, "sum": 0.014321291175807441, "min": 0.014321291175807441}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539441.559388, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1518539441.559371}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.014324195084739925, "sum": 0.014324195084739925, "min": 0.014324195084739925}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539441.559448, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1518539441.55943}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.00566765322847419, "sum": 0.00566765322847419, "min": 0.00566765322847419}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539481.715348, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1518539481.715245}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004825514521464765, "sum": 0.004825514521464765, "min": 0.004825514521464765}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539481.715447, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1518539481.715424}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005666983569392478, "sum": 0.005666983569392478, "min": 0.005666983569392478}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539481.715605, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1518539481.71558}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0048245666384786725, "sum": 0.0048245666384786725, "min": 0.0048245666384786725}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539481.715668, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1518539481.71565}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0049707523119321695, "sum": 0.0049707523119321695, "min": 0.0049707523119321695}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539481.715737, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1518539481.715717}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0086763582909537, "sum": 0.0086763582909537, "min": 0.0086763582909537}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539481.715806, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1518539481.715786}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.00521647686925137, "sum": 0.00521647686925137, "min": 0.00521647686925137}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539481.71588, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1518539481.715856}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.007678870179668914, "sum": 0.007678870179668914, "min": 0.007678870179668914}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539481.715959, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1518539481.715935}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0061606315728351775, "sum": 0.0061606315728351775, "min": 0.0061606315728351775}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539481.716037, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1518539481.716014}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0053956647170894595, "sum": 0.0053956647170894595, "min": 0.0053956647170894595}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539481.716114, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1518539481.71609}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.006159367543536736, "sum": 0.006159367543536736, "min": 0.006159367543536736}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539481.71619, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1518539481.716168}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.00539418336632381, "sum": 0.00539418336632381, "min": 0.00539418336632381}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539481.716261, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1518539481.71624}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005340294556572554, "sum": 0.005340294556572554, "min": 0.005340294556572554}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539481.716325, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1518539481.716305}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.007432439326473239, "sum": 0.007432439326473239, "min": 0.007432439326473239}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539481.716363, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1518539481.716352}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005339804868883338, "sum": 0.005339804868883338, "min": 0.005339804868883338}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539481.716396, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1518539481.716385}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.007413593718322016, "sum": 0.007413593718322016, "min": 0.007413593718322016}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539481.716428, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1518539481.716417}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012193860524150741, "sum": 0.012193860524150741, "min": 0.012193860524150741}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539481.716475, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1518539481.716456}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011884088968805759, "sum": 0.011884088968805759, "min": 0.011884088968805759}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539481.716539, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1518539481.716517}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012192488100274979, "sum": 0.012192488100274979, "min": 0.012192488100274979}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539481.716593, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1518539481.71658}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011884277911340616, "sum": 0.011884277911340616, "min": 0.011884277911340616}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539481.716658, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1518539481.716635}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011849327047680875, "sum": 0.011849327047680875, "min": 0.011849327047680875}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539481.716706, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1518539481.716693}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011949613040320606, "sum": 0.011949613040320606, "min": 0.011949613040320606}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539481.716765, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1518539481.716749}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011849455941229102, "sum": 0.011849455941229102, "min": 0.011849455941229102}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539481.716825, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1518539481.716805}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011950119394494826, "sum": 0.011950119394494826, "min": 0.011950119394494826}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539481.716898, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1518539481.716876}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.013074348761015629, "sum": 0.013074348761015629, "min": 0.013074348761015629}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539481.716967, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1518539481.716945}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01275176924230703, "sum": 0.01275176924230703, "min": 0.01275176924230703}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539481.717036, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1518539481.717013}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.013074506374906345, "sum": 0.013074506374906345, "min": 0.013074506374906345}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539481.717103, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1518539481.717082}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01275279399591038, "sum": 0.01275279399591038, "min": 0.01275279399591038}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539481.71717, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1518539481.717149}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012768974799483296, "sum": 0.012768974799483296, "min": 0.012768974799483296}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539481.717238, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1518539481.717217}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012907124255076948, "sum": 0.012907124255076948, "min": 0.012907124255076948}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539481.717307, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1518539481.717285}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01276839598781433, "sum": 0.01276839598781433, "min": 0.01276839598781433}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539481.717375, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1518539481.717353}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01290260336623643, "sum": 0.01290260336623643, "min": 0.01290260336623643}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539481.717444, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1518539481.717422}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0049351280024013365, "sum": 0.0049351280024013365, "min": 0.0049351280024013365}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539521.853717, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1518539521.853617}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004272033306152885, "sum": 0.004272033306152885, "min": 0.004272033306152885}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539521.853804, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1518539521.853787}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004935854248941902, "sum": 0.004935854248941902, "min": 0.004935854248941902}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539521.853875, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1518539521.853855}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004272103815163714, "sum": 0.004272103815163714, "min": 0.004272103815163714}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539521.853934, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1518539521.853918}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0044202055972832515, "sum": 0.0044202055972832515, "min": 0.0044202055972832515}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539521.854, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1518539521.853979}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.007794465139869357, "sum": 0.007794465139869357, "min": 0.007794465139869357}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539521.854069, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1518539521.854049}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004565744502974831, "sum": 0.004565744502974831, "min": 0.004565744502974831}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539521.854134, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1518539521.854114}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.008612318923995804, "sum": 0.008612318923995804, "min": 0.008612318923995804}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539521.854201, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1518539521.85418}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005545444291638562, "sum": 0.005545444291638562, "min": 0.005545444291638562}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539521.854271, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1518539521.85425}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0050756078201484485, "sum": 0.0050756078201484485, "min": 0.0050756078201484485}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539521.854343, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1518539521.854321}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005544711826256959, "sum": 0.005544711826256959, "min": 0.005544711826256959}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539521.854416, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1518539521.854394}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005074913631376612, "sum": 0.005074913631376612, "min": 0.005074913631376612}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539521.854466, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1518539521.85445}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005148838016589782, "sum": 0.005148838016589782, "min": 0.005148838016589782}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539521.854526, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1518539521.854506}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.007153346891562645, "sum": 0.007153346891562645, "min": 0.007153346891562645}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539521.854594, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1518539521.854572}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005148872248229971, "sum": 0.005148872248229971, "min": 0.005148872248229971}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539521.854664, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1518539521.854641}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.007180838037523475, "sum": 0.007180838037523475, "min": 0.007180838037523475}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539521.854731, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1518539521.854709}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011892645097670546, "sum": 0.011892645097670546, "min": 0.011892645097670546}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539521.854797, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1518539521.854776}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011835506676310994, "sum": 0.011835506676310994, "min": 0.011835506676310994}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539521.854864, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1518539521.854842}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011892130361742763, "sum": 0.011892130361742763, "min": 0.011892130361742763}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539521.854932, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1518539521.854911}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01183551148193548, "sum": 0.01183551148193548, "min": 0.01183551148193548}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539521.854998, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1518539521.854977}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011886852898870606, "sum": 0.011886852898870606, "min": 0.011886852898870606}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539521.855064, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1518539521.855042}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012027238453345666, "sum": 0.012027238453345666, "min": 0.012027238453345666}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539521.855129, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1518539521.855109}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011887565559621557, "sum": 0.011887565559621557, "min": 0.011887565559621557}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539521.855194, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1518539521.855173}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012028014249688232, "sum": 0.012028014249688232, "min": 0.012028014249688232}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539521.855249, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1518539521.855229}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012809304200621494, "sum": 0.012809304200621494, "min": 0.012809304200621494}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539521.855323, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1518539521.8553}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012705335508598143, "sum": 0.012705335508598143, "min": 0.012705335508598143}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539521.855395, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1518539521.855373}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012809425595564297, "sum": 0.012809425595564297, "min": 0.012809425595564297}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539521.855472, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1518539521.855449}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01270543347804781, "sum": 0.01270543347804781, "min": 0.01270543347804781}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539521.85557, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1518539521.855546}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012745337937042177, "sum": 0.012745337937042177, "min": 0.012745337937042177}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539521.855648, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1518539521.855625}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.013007542119402603, "sum": 0.013007542119402603, "min": 0.013007542119402603}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539521.855744, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1518539521.855702}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012743747416868746, "sum": 0.012743747416868746, "min": 0.012743747416868746}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539521.855827, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1518539521.855803}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.013003461209526024, "sum": 0.013003461209526024, "min": 0.013003461209526024}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539521.855902, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1518539521.85588}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0047066798815167095, "sum": 0.0047066798815167095, "min": 0.0047066798815167095}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539562.149102, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1518539562.148997}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004219565435413675, "sum": 0.004219565435413675, "min": 0.004219565435413675}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539562.149195, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1518539562.149178}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004706364447123794, "sum": 0.004706364447123794, "min": 0.004706364447123794}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539562.14927, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1518539562.149249}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004219576283788167, "sum": 0.004219576283788167, "min": 0.004219576283788167}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539562.149347, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1518539562.149325}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004176341185297623, "sum": 0.004176341185297623, "min": 0.004176341185297623}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539562.149401, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1518539562.149382}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.007171034751823812, "sum": 0.007171034751823812, "min": 0.007171034751823812}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539562.149457, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1518539562.149437}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004113307054917586, "sum": 0.004113307054917586, "min": 0.004113307054917586}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539562.149512, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1518539562.149494}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.007403552181039045, "sum": 0.007403552181039045, "min": 0.007403552181039045}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539562.149561, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1518539562.149543}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0053650349981514325, "sum": 0.0053650349981514325, "min": 0.0053650349981514325}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539562.149632, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1518539562.14961}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005026118016790554, "sum": 0.005026118016790554, "min": 0.005026118016790554}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539562.149691, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1518539562.149672}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005364482481346791, "sum": 0.005364482481346791, "min": 0.005364482481346791}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539562.14974, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1518539562.149722}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005025832397960515, "sum": 0.005025832397960515, "min": 0.005025832397960515}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539562.149799, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1518539562.149779}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005029672572770751, "sum": 0.005029672572770751, "min": 0.005029672572770751}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539562.149865, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1518539562.149841}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.007021659353294926, "sum": 0.007021659353294926, "min": 0.007021659353294926}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539562.149931, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1518539562.149912}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0050295735341328454, "sum": 0.0050295735341328454, "min": 0.0050295735341328454}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539562.149982, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1518539562.149964}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.007024515124449757, "sum": 0.007024515124449757, "min": 0.007024515124449757}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539562.150032, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1518539562.150014}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011823025378417298, "sum": 0.011823025378417298, "min": 0.011823025378417298}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539562.150097, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1518539562.150078}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011836684688506835, "sum": 0.011836684688506835, "min": 0.011836684688506835}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539562.150149, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1518539562.15013}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011822717457590812, "sum": 0.011822717457590812, "min": 0.011822717457590812}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539562.1502, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1518539562.15018}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011836674498266963, "sum": 0.011836674498266963, "min": 0.011836674498266963}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539562.150264, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1518539562.150243}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011847777896856208, "sum": 0.011847777896856208, "min": 0.011847777896856208}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539562.150317, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1518539562.150299}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012101646671464374, "sum": 0.012101646671464374, "min": 0.012101646671464374}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539562.150429, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1518539562.150401}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01184803616749235, "sum": 0.01184803616749235, "min": 0.01184803616749235}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539562.150503, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1518539562.150482}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012102348698232039, "sum": 0.012102348698232039, "min": 0.012102348698232039}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539562.150579, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1518539562.150557}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012757100952287994, "sum": 0.012757100952287994, "min": 0.012757100952287994}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539562.150647, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1518539562.150626}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012708294433223674, "sum": 0.012708294433223674, "min": 0.012708294433223674}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539562.15071, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1518539562.15069}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012756788369911502, "sum": 0.012756788369911502, "min": 0.012756788369911502}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539562.150768, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1518539562.150747}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01270806401175069, "sum": 0.01270806401175069, "min": 0.01270806401175069}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539562.150878, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1518539562.150814}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012723198213042264, "sum": 0.012723198213042264, "min": 0.012723198213042264}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539562.150954, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1518539562.150932}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.013092452142982328, "sum": 0.013092452142982328, "min": 0.013092452142982328}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539562.151029, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1518539562.151007}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012723153411863319, "sum": 0.012723153411863319, "min": 0.012723153411863319}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539562.151106, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1518539562.151083}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01308869569287467, "sum": 0.01308869569287467, "min": 0.01308869569287467}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539562.151175, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1518539562.151153}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004611711336905698, "sum": 0.004611711336905698, "min": 0.004611711336905698}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539602.438909, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1518539602.438813}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0041053236841811355, "sum": 0.0041053236841811355, "min": 0.0041053236841811355}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539602.438997, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1518539602.43898}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004597923562648785, "sum": 0.004597923562648785, "min": 0.004597923562648785}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539602.439069, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1518539602.439048}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004105155163942032, "sum": 0.004105155163942032, "min": 0.004105155163942032}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539602.439126, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1518539602.439108}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.003976653361838894, "sum": 0.003976653361838894, "min": 0.003976653361838894}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539602.439188, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1518539602.439169}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.007876232851404532, "sum": 0.007876232851404532, "min": 0.007876232851404532}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539602.439257, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1518539602.439237}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.003935665231044722, "sum": 0.003935665231044722, "min": 0.003935665231044722}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539602.439323, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1518539602.439303}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.006609595590171705, "sum": 0.006609595590171705, "min": 0.006609595590171705}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539602.439398, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1518539602.439375}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005293916314092266, "sum": 0.005293916314092266, "min": 0.005293916314092266}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539602.439466, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1518539602.439443}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005014482119820084, "sum": 0.005014482119820084, "min": 0.005014482119820084}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539602.439538, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1518539602.439523}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.00528608185205474, "sum": 0.00528608185205474, "min": 0.00528608185205474}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539602.439602, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1518539602.439585}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005014375496850195, "sum": 0.005014375496850195, "min": 0.005014375496850195}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539602.439662, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1518539602.439642}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004961481075060655, "sum": 0.004961481075060655, "min": 0.004961481075060655}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539602.439726, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1518539602.439706}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.006869400203925554, "sum": 0.006869400203925554, "min": 0.006869400203925554}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539602.439797, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1518539602.439775}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0049614495638174464, "sum": 0.0049614495638174464, "min": 0.0049614495638174464}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539602.439869, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1518539602.439844}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.006865837064681081, "sum": 0.006865837064681081, "min": 0.006865837064681081}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539602.439925, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1518539602.439905}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011805669935233142, "sum": 0.011805669935233142, "min": 0.011805669935233142}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539602.43998, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1518539602.43996}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011839254112038986, "sum": 0.011839254112038986, "min": 0.011839254112038986}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539602.440044, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1518539602.440023}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011803128637642745, "sum": 0.011803128637642745, "min": 0.011803128637642745}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539602.440111, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1518539602.44009}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011839242469833557, "sum": 0.011839242469833557, "min": 0.011839242469833557}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539602.44017, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1518539602.44015}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0118292998060225, "sum": 0.0118292998060225, "min": 0.0118292998060225}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539602.440239, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1518539602.440218}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01215635249077017, "sum": 0.01215635249077017, "min": 0.01215635249077017}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539602.440302, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1518539602.440282}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011829381193890869, "sum": 0.011829381193890869, "min": 0.011829381193890869}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539602.440355, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1518539602.440335}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012156713432351867, "sum": 0.012156713432351867, "min": 0.012156713432351867}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539602.440418, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1518539602.440398}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012739559562211056, "sum": 0.012739559562211056, "min": 0.012739559562211056}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539602.440471, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1518539602.440451}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012712230465737691, "sum": 0.012712230465737691, "min": 0.012712230465737691}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539602.440524, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1518539602.440505}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012737001888453961, "sum": 0.012737001888453961, "min": 0.012737001888453961}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539602.440591, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1518539602.440569}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012712316856715812, "sum": 0.012712316856715812, "min": 0.012712316856715812}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539602.440661, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1518539602.440641}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012713815661887807, "sum": 0.012713815661887807, "min": 0.012713815661887807}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539602.440722, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1518539602.440702}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.013131031471241011, "sum": 0.013131031471241011, "min": 0.013131031471241011}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539602.440786, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1518539602.440766}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012713467098742604, "sum": 0.012713467098742604, "min": 0.012713467098742604}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539602.440857, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1518539602.440835}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.013128724242869136, "sum": 0.013128724242869136, "min": 0.013128724242869136}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539602.440934, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1518539602.440911}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004567882297149145, "sum": 0.004567882297149145, "min": 0.004567882297149145}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539642.867526, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1518539642.867337}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004108989604124163, "sum": 0.004108989604124163, "min": 0.004108989604124163}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539642.867653, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1518539642.86763}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004496083196521883, "sum": 0.004496083196521883, "min": 0.004496083196521883}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539642.867719, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1518539642.867701}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004109521880671083, "sum": 0.004109521880671083, "min": 0.004109521880671083}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539642.867785, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1518539642.867766}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.003910596923992397, "sum": 0.003910596923992397, "min": 0.003910596923992397}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539642.867845, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1518539642.867827}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0063861206774861665, "sum": 0.0063861206774861665, "min": 0.0063861206774861665}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539642.867906, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1518539642.867888}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0038892724012851297, "sum": 0.0038892724012851297, "min": 0.0038892724012851297}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539642.867967, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1518539642.867949}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.007738234744593869, "sum": 0.007738234744593869, "min": 0.007738234744593869}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539642.868021, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1518539642.868004}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005262416164020458, "sum": 0.005262416164020458, "min": 0.005262416164020458}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539642.868074, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1518539642.868058}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005009080017187509, "sum": 0.005009080017187509, "min": 0.005009080017187509}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539642.86813, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1518539642.868113}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005213723326453483, "sum": 0.005213723326453483, "min": 0.005213723326453483}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539642.86819, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1518539642.868173}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005009045735914186, "sum": 0.005009045735914186, "min": 0.005009045735914186}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539642.868248, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1518539642.868231}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004925963872048749, "sum": 0.004925963872048749, "min": 0.004925963872048749}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539642.868304, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1518539642.868287}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.006781071410121688, "sum": 0.006781071410121688, "min": 0.006781071410121688}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539642.868357, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1518539642.86834}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004925954096976773, "sum": 0.004925954096976773, "min": 0.004925954096976773}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539642.868413, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1518539642.868396}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0067824655376168075, "sum": 0.0067824655376168075, "min": 0.0067824655376168075}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539642.868469, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1518539642.868452}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011795597379376372, "sum": 0.011795597379376372, "min": 0.011795597379376372}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539642.86853, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1518539642.868512}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011842862434804919, "sum": 0.011842862434804919, "min": 0.011842862434804919}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539642.868591, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1518539642.868573}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011787788653559235, "sum": 0.011787788653559235, "min": 0.011787788653559235}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539642.868652, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1518539642.868635}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01184285222466691, "sum": 0.01184285222466691, "min": 0.01184285222466691}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539642.868713, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1518539642.868696}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011788364326259697, "sum": 0.011788364326259697, "min": 0.011788364326259697}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539642.868771, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1518539642.868754}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012187641194037884, "sum": 0.012187641194037884, "min": 0.012187641194037884}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539642.868831, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1518539642.868814}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011788384220506772, "sum": 0.011788384220506772, "min": 0.011788384220506772}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539642.868889, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1518539642.868872}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01218779655917731, "sum": 0.01218779655917731, "min": 0.01218779655917731}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539642.868943, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1518539642.868926}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012732437036512606, "sum": 0.012732437036512606, "min": 0.012732437036512606}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539642.868996, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1518539642.868979}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012717095234966183, "sum": 0.012717095234966183, "min": 0.012717095234966183}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539642.869052, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1518539642.869035}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012721729587359601, "sum": 0.012721729587359601, "min": 0.012721729587359601}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539642.869107, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1518539642.869091}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012716884811120818, "sum": 0.012716884811120818, "min": 0.012716884811120818}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539642.869161, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1518539642.869144}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012706778089714097, "sum": 0.012706778089714097, "min": 0.012706778089714097}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539642.869216, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1518539642.8692}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01312382613607948, "sum": 0.01312382613607948, "min": 0.01312382613607948}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539642.869277, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1518539642.86926}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012706586411260217, "sum": 0.012706586411260217, "min": 0.012706586411260217}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539642.869338, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1518539642.869321}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.013123198056230733, "sum": 0.013123198056230733, "min": 0.013123198056230733}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539642.8694, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1518539642.869383}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004545868605704432, "sum": 0.004545868605704432, "min": 0.004545868605704432}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539683.320407, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1518539683.32026}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.00404934419876331, "sum": 0.00404934419876331, "min": 0.00404934419876331}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539683.320526, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1518539683.320504}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0043974109476499525, "sum": 0.0043974109476499525, "min": 0.0043974109476499525}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539683.320594, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1518539683.320576}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004049361773287825, "sum": 0.004049361773287825, "min": 0.004049361773287825}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539683.320661, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1518539683.320642}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0038826590647778837, "sum": 0.0038826590647778837, "min": 0.0038826590647778837}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539683.320725, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1518539683.320707}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.007831224101185556, "sum": 0.007831224101185556, "min": 0.007831224101185556}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539683.320785, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1518539683.320767}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.003856997669124639, "sum": 0.003856997669124639, "min": 0.003856997669124639}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539683.320839, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1518539683.320822}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.00675714918727209, "sum": 0.00675714918727209, "min": 0.00675714918727209}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539683.320893, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1518539683.320877}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005247287381604972, "sum": 0.005247287381604972, "min": 0.005247287381604972}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539683.320947, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1518539683.32093}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005004265091520356, "sum": 0.005004265091520356, "min": 0.005004265091520356}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539683.321001, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1518539683.320984}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005145370338680634, "sum": 0.005145370338680634, "min": 0.005145370338680634}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539683.321057, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1518539683.32104}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.00500425663870202, "sum": 0.00500425663870202, "min": 0.00500425663870202}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539683.321118, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1518539683.3211}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004907966149619305, "sum": 0.004907966149619305, "min": 0.004907966149619305}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539683.32118, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1518539683.321162}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0066095231342887745, "sum": 0.0066095231342887745, "min": 0.0066095231342887745}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539683.321241, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1518539683.321224}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004907936201501922, "sum": 0.004907936201501922, "min": 0.004907936201501922}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539683.321302, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1518539683.321285}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.006611414136821835, "sum": 0.006611414136821835, "min": 0.006611414136821835}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539683.321364, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1518539683.321346}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011789075167153973, "sum": 0.011789075167153973, "min": 0.011789075167153973}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539683.321425, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1518539683.321408}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011846058535498069, "sum": 0.011846058535498069, "min": 0.011846058535498069}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539683.321483, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1518539683.321466}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011777278517773113, "sum": 0.011777278517773113, "min": 0.011777278517773113}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539683.321544, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1518539683.321526}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011846051187101138, "sum": 0.011846051187101138, "min": 0.011846051187101138}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539683.3216, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1518539683.321583}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01176927949904079, "sum": 0.01176927949904079, "min": 0.01176927949904079}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539683.321657, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1518539683.32164}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012179446821400229, "sum": 0.012179446821400229, "min": 0.012179446821400229}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539683.32171, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1518539683.321693}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01176929975624185, "sum": 0.01176929975624185, "min": 0.01176929975624185}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539683.321765, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1518539683.321748}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012179503324030184, "sum": 0.012179503324030184, "min": 0.012179503324030184}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539683.32182, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1518539683.321804}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012728928184832435, "sum": 0.012728928184832435, "min": 0.012728928184832435}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539683.321875, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1518539683.321859}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012720759155221254, "sum": 0.012720759155221254, "min": 0.012720759155221254}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539683.321932, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1518539683.321915}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012711438694632197, "sum": 0.012711438694632197, "min": 0.012711438694632197}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539683.321988, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1518539683.321971}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012720677681626326, "sum": 0.012720677681626326, "min": 0.012720677681626326}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539683.322043, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1518539683.322027}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012704518948902328, "sum": 0.012704518948902328, "min": 0.012704518948902328}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539683.322099, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1518539683.322082}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.013100453330518073, "sum": 0.013100453330518073, "min": 0.013100453330518073}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539683.322159, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1518539683.322142}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012704512404958646, "sum": 0.012704512404958646, "min": 0.012704512404958646}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539683.322242, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1518539683.32222}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.013100190397752186, "sum": 0.013100190397752186, "min": 0.013100190397752186}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539683.322302, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1518539683.322283}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004526677778192672, "sum": 0.004526677778192672, "min": 0.004526677778192672}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539724.920011, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1518539724.919915}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0040510251828241185, "sum": 0.0040510251828241185, "min": 0.0040510251828241185}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539724.92011, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1518539724.920087}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0043158132050291126, "sum": 0.0043158132050291126, "min": 0.0043158132050291126}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539724.920243, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1518539724.92022}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004050180944595411, "sum": 0.004050180944595411, "min": 0.004050180944595411}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539724.920299, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1518539724.920279}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.003871279326809877, "sum": 0.003871279326809877, "min": 0.003871279326809877}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539724.920383, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1518539724.920363}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005999574514424305, "sum": 0.005999574514424305, "min": 0.005999574514424305}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539724.920439, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1518539724.92042}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.003850509470541883, "sum": 0.003850509470541883, "min": 0.003850509470541883}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539724.92055, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1518539724.92052}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.006178274866843752, "sum": 0.006178274866843752, "min": 0.006178274866843752}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539724.920626, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1518539724.920604}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005234769326017564, "sum": 0.005234769326017564, "min": 0.005234769326017564}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539724.920707, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1518539724.920686}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004998800142043566, "sum": 0.004998800142043566, "min": 0.004998800142043566}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539724.920761, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1518539724.920741}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0050895890555378185, "sum": 0.0050895890555378185, "min": 0.0050895890555378185}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539724.920813, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1518539724.920794}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004998799308416355, "sum": 0.004998799308416355, "min": 0.004998799308416355}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539724.920878, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1518539724.920857}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004900847045876774, "sum": 0.004900847045876774, "min": 0.004900847045876774}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539724.920931, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1518539724.920912}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.006530738546944085, "sum": 0.006530738546944085, "min": 0.006530738546944085}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539724.92099, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1518539724.920969}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004905023857068465, "sum": 0.004905023857068465, "min": 0.004905023857068465}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539724.921051, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1518539724.92103}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.006530864075280681, "sum": 0.006530864075280681, "min": 0.006530864075280681}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539724.921104, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1518539724.921084}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011786108992724533, "sum": 0.011786108992724533, "min": 0.011786108992724533}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539724.921155, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1518539724.921135}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011847347252549656, "sum": 0.011847347252549656, "min": 0.011847347252549656}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539724.921213, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1518539724.921193}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01177206226486159, "sum": 0.01177206226486159, "min": 0.01177206226486159}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539724.921278, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1518539724.921258}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011847343217267329, "sum": 0.011847343217267329, "min": 0.011847343217267329}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539724.921332, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1518539724.921312}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011768166285352296, "sum": 0.011768166285352296, "min": 0.011768166285352296}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539724.921393, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1518539724.921373}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012139096862980609, "sum": 0.012139096862980609, "min": 0.012139096862980609}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539724.921444, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1518539724.921425}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01176890274337736, "sum": 0.01176890274337736, "min": 0.01176890274337736}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539724.921499, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1518539724.921479}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012139115288150478, "sum": 0.012139115288150478, "min": 0.012139115288150478}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539724.921559, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1518539724.921539}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012726223472520769, "sum": 0.012726223472520769, "min": 0.012726223472520769}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539724.921623, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1518539724.921602}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012722106522854312, "sum": 0.012722106522854312, "min": 0.012722106522854312}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539724.921689, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1518539724.921668}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012706393057173753, "sum": 0.012706393057173753, "min": 0.012706393057173753}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539724.921764, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1518539724.921741}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01272216352099276, "sum": 0.01272216352099276, "min": 0.01272216352099276}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539724.921837, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1518539724.921815}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012702869320490274, "sum": 0.012702869320490274, "min": 0.012702869320490274}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539724.921908, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1518539724.921886}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.013068543827931685, "sum": 0.013068543827931685, "min": 0.013068543827931685}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539724.921979, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1518539724.921957}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01270383550385275, "sum": 0.01270383550385275, "min": 0.01270383550385275}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539724.922054, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1518539724.922031}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.013068431260655204, "sum": 0.013068431260655204, "min": 0.013068431260655204}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539724.922124, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1518539724.922103}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004505496464639783, "sum": 0.004505496464639783, "min": 0.004505496464639783}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539765.396218, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1518539765.396114}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004010293117862748, "sum": 0.004010293117862748, "min": 0.004010293117862748}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539765.396306, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1518539765.396289}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0042553444329394875, "sum": 0.0042553444329394875, "min": 0.0042553444329394875}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539765.396379, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1518539765.396359}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004009794285090782, "sum": 0.004009794285090782, "min": 0.004009794285090782}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539765.396436, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1518539765.396417}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.00386691662254762, "sum": 0.00386691662254762, "min": 0.00386691662254762}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539765.396499, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1518539765.396478}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.006915447515173465, "sum": 0.006915447515173465, "min": 0.006915447515173465}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539765.396558, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1518539765.396539}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.003849467492314527, "sum": 0.003849467492314527, "min": 0.003849467492314527}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539765.396615, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1518539765.396596}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.006020300454856551, "sum": 0.006020300454856551, "min": 0.006020300454856551}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539765.396681, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1518539765.39666}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005221268920355054, "sum": 0.005221268920355054, "min": 0.005221268920355054}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539765.396748, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1518539765.396727}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004992413661222022, "sum": 0.004992413661222022, "min": 0.004992413661222022}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539765.396814, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1518539765.396792}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005048058589164871, "sum": 0.005048058589164871, "min": 0.005048058589164871}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539765.396868, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1518539765.396848}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004992413620827308, "sum": 0.004992413620827308, "min": 0.004992413620827308}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539765.396921, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1518539765.396902}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004898119441207394, "sum": 0.004898119441207394, "min": 0.004898119441207394}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539765.396988, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1518539765.396966}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.006300828574878916, "sum": 0.006300828574878916, "min": 0.006300828574878916}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539765.397055, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1518539765.397034}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0049045469777950325, "sum": 0.0049045469777950325, "min": 0.0049045469777950325}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539765.397121, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1518539765.3971}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.006300838523816474, "sum": 0.006300838523816474, "min": 0.006300838523816474}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539765.397188, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1518539765.397167}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011783242223851652, "sum": 0.011783242223851652, "min": 0.011783242223851652}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539765.397252, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1518539765.397231}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011846591975973791, "sum": 0.011846591975973791, "min": 0.011846591975973791}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539765.39732, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1518539765.397299}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01177007754090201, "sum": 0.01177007754090201, "min": 0.01177007754090201}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539765.397374, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1518539765.397354}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011846590325924528, "sum": 0.011846590325924528, "min": 0.011846590325924528}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539765.397426, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1518539765.397407}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011764718600993894, "sum": 0.011764718600993894, "min": 0.011764718600993894}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539765.397494, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1518539765.397473}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012089582446618492, "sum": 0.012089582446618492, "min": 0.012089582446618492}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539765.397547, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1518539765.397527}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011769188991990913, "sum": 0.011769188991990913, "min": 0.011769188991990913}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539765.397598, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1518539765.397578}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01208958860191266, "sum": 0.01208958860191266, "min": 0.01208958860191266}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539765.397654, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1518539765.397636}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012723508118894923, "sum": 0.012723508118894923, "min": 0.012723508118894923}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539765.397703, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1518539765.397685}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012721733193859518, "sum": 0.012721733193859518, "min": 0.012721733193859518}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539765.397753, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1518539765.397734}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012704492828034493, "sum": 0.012704492828034493, "min": 0.012704492828034493}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539765.397812, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1518539765.397793}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012721765919412714, "sum": 0.012721765919412714, "min": 0.012721765919412714}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539765.397861, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1518539765.397843}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012702389768059234, "sum": 0.012702389768059234, "min": 0.012702389768059234}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539765.397911, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1518539765.397892}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.013035495161587545, "sum": 0.013035495161587545, "min": 0.013035495161587545}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539765.397983, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1518539765.397961}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01270381375653557, "sum": 0.01270381375653557, "min": 0.01270381375653557}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539765.39804, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1518539765.39802}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.013035462793190945, "sum": 0.013035462793190945, "min": 0.013035462793190945}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539765.3981, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1518539765.39808}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004484349691110802, "sum": 0.004484349691110802, "min": 0.004484349691110802}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539805.854036, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1518539805.853937}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.00402259058017942, "sum": 0.00402259058017942, "min": 0.00402259058017942}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539805.854131, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1518539805.854108}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004211591096056334, "sum": 0.004211591096056334, "min": 0.004211591096056334}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539805.854201, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1518539805.854183}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004024087728027359, "sum": 0.004024087728027359, "min": 0.004024087728027359}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539805.854257, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1518539805.854237}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.00386533543815365, "sum": 0.00386533543815365, "min": 0.00386533543815365}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539805.854311, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1518539805.854291}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.005654576864391827, "sum": 0.005654576864391827, "min": 0.005654576864391827}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539805.854378, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1518539805.854357}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0038483884434840345, "sum": 0.0038483884434840345, "min": 0.0038483884434840345}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539805.854431, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1518539805.854412}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.0058446863352930745, "sum": 0.0058446863352930745, "min": 0.0058446863352930745}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539805.854482, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1518539805.854464}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.00520785849070154, "sum": 0.00520785849070154, "min": 0.00520785849070154}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539805.854541, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1518539805.854521}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004985605148845408, "sum": 0.004985605148845408, "min": 0.004985605148845408}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539805.854589, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1518539805.854571}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.00501769044058749, "sum": 0.00501769044058749, "min": 0.00501769044058749}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539805.854649, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1518539805.85463}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004985605566145246, "sum": 0.004985605566145246, "min": 0.004985605566145246}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539805.854712, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1518539805.854692}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.00489719802117252, "sum": 0.00489719802117252, "min": 0.00489719802117252}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539805.854769, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1518539805.854749}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.006017270903420975, "sum": 0.006017270903420975, "min": 0.006017270903420975}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539805.854818, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1518539805.854799}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.004904112131540555, "sum": 0.004904112131540555, "min": 0.004904112131540555}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539805.854884, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1518539805.854863}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.006017276760392699, "sum": 0.006017276760392699, "min": 0.006017276760392699}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539805.85495, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1518539805.854928}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011780606644699373, "sum": 0.011780606644699373, "min": 0.011780606644699373}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539805.855011, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1518539805.854992}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011844644642708053, "sum": 0.011844644642708053, "min": 0.011844644642708053}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539805.855074, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1518539805.855053}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011769445450101272, "sum": 0.011769445450101272, "min": 0.011769445450101272}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539805.855139, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1518539805.855118}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011844643967966717, "sum": 0.011844643967966717, "min": 0.011844643967966717}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539805.855209, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1518539805.855188}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011761395070686877, "sum": 0.011761395070686877, "min": 0.011761395070686877}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539805.855279, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1518539805.855257}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012047038052163749, "sum": 0.012047038052163749, "min": 0.012047038052163749}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539805.855346, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1518539805.855324}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.011769212986899907, "sum": 0.011769212986899907, "min": 0.011769212986899907}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539805.85541, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1518539805.85539}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01204704018556569, "sum": 0.01204704018556569, "min": 0.01204704018556569}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539805.855476, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1518539805.855455}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012720975071550851, "sum": 0.012720975071550851, "min": 0.012720975071550851}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539805.855565, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1518539805.855543}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012720586996091656, "sum": 0.012720586996091656, "min": 0.012720586996091656}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539805.855628, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1518539805.855608}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01270391387992116, "sum": 0.01270391387992116, "min": 0.01270391387992116}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539805.855681, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1518539805.855661}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012720593902540494, "sum": 0.012720593902540494, "min": 0.012720593902540494}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539805.855747, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1518539805.855725}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.01270207732137906, "sum": 0.01270207732137906, "min": 0.01270207732137906}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539805.855817, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1518539805.855794}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.013001564730736746, "sum": 0.013001564730736746, "min": 0.013001564730736746}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539805.855885, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1518539805.855863}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.012703782003746934, "sum": 0.012703782003746934, "min": 0.012703782003746934}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539805.855951, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1518539805.85593}
    [0m
    [31m#metrics {"Metrics": {"training_binary_classification_cross_entropy": {"count": 1, "max": 0.013001554601836994, "sum": 0.013001554601836994, "min": 0.013001554601836994}, "validation_binary_classification_cross_entropy": {"count": 1, "max": -Infinity, "sum": NaN, "min": Infinity}}, "EndTime": 1518539805.856015, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1518539805.855995}
    [0m
    [31m[02/13/2018 16:36:45 WARNING 140471961589568] wait_for_all_workers will not sync workers since the kv store is not running distributed[0m
    [31m[02/13/2018 16:36:45 INFO 140471961589568] Update[9961]: now learning rate arrived at 1.00000e-05, will not change in the future[0m
    [31m[02/13/2018 16:36:45 INFO 140471961589568] Update[9961]: now learning rate arrived at 1.00000e-04, will not change in the future[0m
    [31m[02/13/2018 16:36:45 INFO 140471961589568] Update[9961]: now learning rate arrived at 1.00000e-05, will not change in the future[0m
    [31m[02/13/2018 16:36:45 INFO 140471961589568] Update[9961]: now learning rate arrived at 1.00000e-04, will not change in the future[0m
    [31m[02/13/2018 16:36:45 INFO 140471961589568] Update[9961]: now learning rate arrived at 1.00000e-05, will not change in the future[0m
    [31m[02/13/2018 16:36:45 INFO 140471961589568] Update[9961]: now learning rate arrived at 1.00000e-04, will not change in the future[0m
    [31m[02/13/2018 16:36:45 INFO 140471961589568] Update[9961]: now learning rate arrived at 1.00000e-05, will not change in the future[0m
    [31m[02/13/2018 16:36:45 INFO 140471961589568] Update[9961]: now learning rate arrived at 1.00000e-04, will not change in the future[0m
    [31m[02/13/2018 16:36:45 INFO 140471961589568] Update[9961]: now learning rate arrived at 1.00000e-05, will not change in the future[0m
    [31m[02/13/2018 16:36:45 INFO 140471961589568] Update[9961]: now learning rate arrived at 1.00000e-04, will not change in the future[0m
    [31m[02/13/2018 16:36:45 INFO 140471961589568] Update[9961]: now learning rate arrived at 1.00000e-05, will not change in the future[0m
    [31m[02/13/2018 16:36:45 INFO 140471961589568] Update[9961]: now learning rate arrived at 1.00000e-04, will not change in the future[0m
    [31m[02/13/2018 16:36:45 INFO 140471961589568] Update[9961]: now learning rate arrived at 1.00000e-05, will not change in the future[0m
    [31m[02/13/2018 16:36:45 INFO 140471961589568] Update[9961]: now learning rate arrived at 1.00000e-04, will not change in the future[0m
    [31m[02/13/2018 16:36:45 INFO 140471961589568] Update[9961]: now learning rate arrived at 1.00000e-05, will not change in the future[0m
    [31m[02/13/2018 16:36:45 INFO 140471961589568] Update[9961]: now learning rate arrived at 1.00000e-04, will not change in the future[0m
    [31m#metrics {"Metrics": {"f1": {"count": 1, "max": 0.8033707865168539, "sum": 0.8033707865168539, "min": 0.8033707865168539}, "threshold_for_accuracy": {"count": 1, "max": 0.12957409024238586, "sum": 0.12957409024238586, "min": 0.12957409024238586}, "threshold_for_precision_at_target_recall": {"count": 1, "max": 0.06492825597524643, "sum": 0.06492825597524643, "min": 0.06492825597524643}, "threshold_for_recall_at_target_precision": {"count": 1, "max": 0.08203981071710587, "sum": 0.08203981071710587, "min": 0.08203981071710587}, "recall_at_precision": {"count": 1, "max": 0.7971014492754211, "sum": 0.7971014492754211, "min": 0.7971014492754211}, "precision_at_target_recall": {"count": 1, "max": 0.798270893371816, "sum": 0.798270893371816, "min": 0.798270893371816}, "accuracy": {"count": 1, "max": 0.999332878553801, "sum": 0.999332878553801, "min": 0.999332878553801}, "threshold_for_f1": {"count": 1, "max": 0.038537416607141495, "sum": 0.038537416607141495, "min": 0.038537416607141495}}, "EndTime": 1518539812.888635, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner"}, "StartTime": 1518539812.879209}
    [0m
    [31m#metrics {"Metrics": {"f1": {"count": 1, "max": 0.8, "sum": 0.8, "min": 0.8}, "threshold_for_accuracy": {"count": 1, "max": 0.10486023128032684, "sum": 0.10486023128032684, "min": 0.10486023128032684}, "threshold_for_precision_at_target_recall": {"count": 1, "max": 0.06992615014314651, "sum": 0.06992615014314651, "min": 0.06992615014314651}, "threshold_for_recall_at_target_precision": {"count": 1, "max": 0.10486023128032684, "sum": 0.10486023128032684, "min": 0.10486023128032684}, "recall_at_precision": {"count": 1, "max": 0.7826086956522369, "sum": 0.7826086956522369, "min": 0.7826086956522369}, "precision_at_target_recall": {"count": 1, "max": 0.789772727272787, "sum": 0.789772727272787, "min": 0.789772727272787}, "accuracy": {"count": 1, "max": 0.9993027828494613, "sum": 0.9993027828494613, "min": 0.9993027828494613}, "threshold_for_f1": {"count": 1, "max": 0.05500476807355881, "sum": 0.05500476807355881, "min": 0.05500476807355881}}, "EndTime": 1518539812.898937, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner"}, "StartTime": 1518539812.888759}
    [0m
    [31m#metrics {"Metrics": {"f1": {"count": 1, "max": 0.7982583454281568, "sum": 0.7982583454281568, "min": 0.7982583454281568}, "threshold_for_accuracy": {"count": 1, "max": 0.06907900422811508, "sum": 0.06907900422811508, "min": 0.06907900422811508}, "threshold_for_precision_at_target_recall": {"count": 1, "max": 0.05102936923503876, "sum": 0.05102936923503876, "min": 0.05102936923503876}, "threshold_for_recall_at_target_precision": {"count": 1, "max": 0.1133095920085907, "sum": 0.1133095920085907, "min": 0.1133095920085907}, "recall_at_precision": {"count": 1, "max": 0.7594202898551422, "sum": 0.7594202898551422, "min": 0.7594202898551422}, "precision_at_target_recall": {"count": 1, "max": 0.789772727272787, "sum": 0.789772727272787, "min": 0.789772727272787}, "accuracy": {"count": 1, "max": 0.9993027828494613, "sum": 0.9993027828494613, "min": 0.9993027828494613}, "threshold_for_f1": {"count": 1, "max": 0.06907900422811508, "sum": 0.06907900422811508, "min": 0.06907900422811508}}, "EndTime": 1518539812.909196, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner"}, "StartTime": 1518539812.899035}
    [0m
    [31m#metrics {"Metrics": {"f1": {"count": 1, "max": 0.8, "sum": 0.8, "min": 0.8}, "threshold_for_accuracy": {"count": 1, "max": 0.10471092909574509, "sum": 0.10471092909574509, "min": 0.10471092909574509}, "threshold_for_precision_at_target_recall": {"count": 1, "max": 0.07017122954130173, "sum": 0.07017122954130173, "min": 0.07017122954130173}, "threshold_for_recall_at_target_precision": {"count": 1, "max": 0.10471092909574509, "sum": 0.10471092909574509, "min": 0.10471092909574509}, "recall_at_precision": {"count": 1, "max": 0.7826086956522369, "sum": 0.7826086956522369, "min": 0.7826086956522369}, "precision_at_target_recall": {"count": 1, "max": 0.789772727272787, "sum": 0.789772727272787, "min": 0.789772727272787}, "accuracy": {"count": 1, "max": 0.9993027828494613, "sum": 0.9993027828494613, "min": 0.9993027828494613}, "threshold_for_f1": {"count": 1, "max": 0.054946549236774445, "sum": 0.054946549236774445, "min": 0.054946549236774445}}, "EndTime": 1518539812.918638, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner"}, "StartTime": 1518539812.909282}
    [0m
    [31m#metrics {"Metrics": {"f1": {"count": 1, "max": 0.7947598253275109, "sum": 0.7947598253275109, "min": 0.7947598253275109}, "threshold_for_accuracy": {"count": 1, "max": 0.11456582695245743, "sum": 0.11456582695245743, "min": 0.11456582695245743}, "threshold_for_precision_at_target_recall": {"count": 1, "max": 0.06263183802366257, "sum": 0.06263183802366257, "min": 0.06263183802366257}, "threshold_for_recall_at_target_precision": {"count": 1, "max": 0.11456582695245743, "sum": 0.11456582695245743, "min": 0.11456582695245743}, "recall_at_precision": {"count": 1, "max": 0.7739130434783263, "sum": 0.7739130434783263, "min": 0.7739130434783263}, "precision_at_target_recall": {"count": 1, "max": 0.7793296089386091, "sum": 0.7793296089386091, "min": 0.7793296089386091}, "accuracy": {"count": 1, "max": 0.9992927509480147, "sum": 0.9992927509480147, "min": 0.9992927509480147}, "threshold_for_f1": {"count": 1, "max": 0.09090258181095123, "sum": 0.09090258181095123, "min": 0.09090258181095123}}, "EndTime": 1518539812.928154, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner"}, "StartTime": 1518539812.918724}
    [0m
    [31m#metrics {"Metrics": {"f1": {"count": 1, "max": 0.8159057437407953, "sum": 0.8159057437407953, "min": 0.8159057437407953}, "threshold_for_accuracy": {"count": 1, "max": 0.49924325942993164, "sum": 0.49924325942993164, "min": 0.49924325942993164}, "threshold_for_precision_at_target_recall": {"count": 1, "max": 0.49924325942993164, "sum": 0.49924325942993164, "min": 0.49924325942993164}, "threshold_for_recall_at_target_precision": {"count": 1, "max": 0.49924325942993164, "sum": 0.49924325942993164, "min": 0.49924325942993164}, "recall_at_precision": {"count": 1, "max": 0.8028985507246948, "sum": 0.8028985507246948, "min": 0.8028985507246948}, "precision_at_target_recall": {"count": 1, "max": 0.8293413173653206, "sum": 0.8293413173653206, "min": 0.8293413173653206}, "accuracy": {"count": 1, "max": 0.9993730061595875, "sum": 0.9993730061595875, "min": 0.9993730061595875}, "threshold_for_f1": {"count": 1, "max": 0.49924325942993164, "sum": 0.49924325942993164, "min": 0.49924325942993164}}, "EndTime": 1518539812.937171, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner"}, "StartTime": 1518539812.928244}
    [0m
    [31m#metrics {"Metrics": {"f1": {"count": 1, "max": 0.7965116279069767, "sum": 0.7965116279069767, "min": 0.7965116279069767}, "threshold_for_accuracy": {"count": 1, "max": 0.09966737776994705, "sum": 0.09966737776994705, "min": 0.09966737776994705}, "threshold_for_precision_at_target_recall": {"count": 1, "max": 0.06558861583471298, "sum": 0.06558861583471298, "min": 0.06558861583471298}, "threshold_for_recall_at_target_precision": {"count": 1, "max": 0.09966737776994705, "sum": 0.09966737776994705, "min": 0.09966737776994705}, "recall_at_precision": {"count": 1, "max": 0.7826086956522369, "sum": 0.7826086956522369, "min": 0.7826086956522369}, "precision_at_target_recall": {"count": 1, "max": 0.7787114845938995, "sum": 0.7787114845938995, "min": 0.7787114845938995}, "accuracy": {"count": 1, "max": 0.9993027828494613, "sum": 0.9993027828494613, "min": 0.9993027828494613}, "threshold_for_f1": {"count": 1, "max": 0.08785610646009445, "sum": 0.08785610646009445, "min": 0.08785610646009445}}, "EndTime": 1518539812.946538, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner"}, "StartTime": 1518539812.937262}
    [0m
    [31m#metrics {"Metrics": {"f1": {"count": 1, "max": 0.8207407407407408, "sum": 0.8207407407407408, "min": 0.8207407407407408}, "threshold_for_accuracy": {"count": 1, "max": 0.42513737082481384, "sum": 0.42513737082481384, "min": 0.42513737082481384}, "threshold_for_precision_at_target_recall": {"count": 1, "max": 0.42513737082481384, "sum": 0.42513737082481384, "min": 0.42513737082481384}, "threshold_for_recall_at_target_precision": {"count": 1, "max": 0.30205410718917847, "sum": 0.30205410718917847, "min": 0.30205410718917847}, "recall_at_precision": {"count": 1, "max": 0.8086956521739684, "sum": 0.8086956521739684, "min": 0.8086956521739684}, "precision_at_target_recall": {"count": 1, "max": 0.8393939393939881, "sum": 0.8393939393939881, "min": 0.8393939393939881}, "accuracy": {"count": 1, "max": 0.9993930699624807, "sum": 0.9993930699624807, "min": 0.9993930699624807}, "threshold_for_f1": {"count": 1, "max": 0.42513737082481384, "sum": 0.42513737082481384, "min": 0.42513737082481384}}, "EndTime": 1518539812.955567, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner"}, "StartTime": 1518539812.946625}
    [0m
    [31m#metrics {"Metrics": {"f1": {"count": 1, "max": 0.8184615384615385, "sum": 0.8184615384615385, "min": 0.8184615384615385}, "threshold_for_accuracy": {"count": 1, "max": 0.03788493573665619, "sum": 0.03788493573665619, "min": 0.03788493573665619}, "threshold_for_precision_at_target_recall": {"count": 1, "max": 0.019479045644402504, "sum": 0.019479045644402504, "min": 0.019479045644402504}, "threshold_for_recall_at_target_precision": {"count": 1, "max": 0.019479045644402504, "sum": 0.019479045644402504, "min": 0.019479045644402504}, "recall_at_precision": {"count": 1, "max": 0.8086956521739684, "sum": 0.8086956521739684, "min": 0.8086956521739684}, "precision_at_target_recall": {"count": 1, "max": 0.8086956521739684, "sum": 0.8086956521739684, "min": 0.8086956521739684}, "accuracy": {"count": 1, "max": 0.9994081178146506, "sum": 0.9994081178146506, "min": 0.9994081178146506}, "threshold_for_f1": {"count": 1, "max": 0.03788493573665619, "sum": 0.03788493573665619, "min": 0.03788493573665619}}, "EndTime": 1518539812.964852, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner"}, "StartTime": 1518539812.955655}
    [0m
    [31m#metrics {"Metrics": {"f1": {"count": 1, "max": 0.8040345821325648, "sum": 0.8040345821325648, "min": 0.8040345821325648}, "threshold_for_accuracy": {"count": 1, "max": 0.06382375210523605, "sum": 0.06382375210523605, "min": 0.06382375210523605}, "threshold_for_precision_at_target_recall": {"count": 1, "max": 0.016658633947372437, "sum": 0.016658633947372437, "min": 0.016658633947372437}, "threshold_for_recall_at_target_precision": {"count": 1, "max": 0.06382375210523605, "sum": 0.06382375210523605, "min": 0.06382375210523605}, "recall_at_precision": {"count": 1, "max": 0.7362318840580475, "sum": 0.7362318840580475, "min": 0.7362318840580475}, "precision_at_target_recall": {"count": 1, "max": 0.7994269340974787, "sum": 0.7994269340974787, "min": 0.7994269340974787}, "accuracy": {"count": 1, "max": 0.9993529423566943, "sum": 0.9993529423566943, "min": 0.9993529423566943}, "threshold_for_f1": {"count": 1, "max": 0.016658633947372437, "sum": 0.016658633947372437, "min": 0.016658633947372437}}, "EndTime": 1518539812.974236, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner"}, "StartTime": 1518539812.964937}
    [0m
    [31m#metrics {"Metrics": {"f1": {"count": 1, "max": 0.8184615384615385, "sum": 0.8184615384615385, "min": 0.8184615384615385}, "threshold_for_accuracy": {"count": 1, "max": 0.03213685378432274, "sum": 0.03213685378432274, "min": 0.03213685378432274}, "threshold_for_precision_at_target_recall": {"count": 1, "max": 0.017181506380438805, "sum": 0.017181506380438805, "min": 0.017181506380438805}, "threshold_for_recall_at_target_precision": {"count": 1, "max": 0.017181506380438805, "sum": 0.017181506380438805, "min": 0.017181506380438805}, "recall_at_precision": {"count": 1, "max": 0.8028985507246948, "sum": 0.8028985507246948, "min": 0.8028985507246948}, "precision_at_target_recall": {"count": 1, "max": 0.8195266272189883, "sum": 0.8195266272189883, "min": 0.8195266272189883}, "accuracy": {"count": 1, "max": 0.9994081178146506, "sum": 0.9994081178146506, "min": 0.9994081178146506}, "threshold_for_f1": {"count": 1, "max": 0.03213685378432274, "sum": 0.03213685378432274, "min": 0.03213685378432274}}, "EndTime": 1518539812.983636, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner"}, "StartTime": 1518539812.974326}
    [0m
    [31m#metrics {"Metrics": {"f1": {"count": 1, "max": 0.8040345821325648, "sum": 0.8040345821325648, "min": 0.8040345821325648}, "threshold_for_accuracy": {"count": 1, "max": 0.06382368505001068, "sum": 0.06382368505001068, "min": 0.06382368505001068}, "threshold_for_precision_at_target_recall": {"count": 1, "max": 0.016658587381243706, "sum": 0.016658587381243706, "min": 0.016658587381243706}, "threshold_for_recall_at_target_precision": {"count": 1, "max": 0.06382368505001068, "sum": 0.06382368505001068, "min": 0.06382368505001068}, "recall_at_precision": {"count": 1, "max": 0.7362318840580475, "sum": 0.7362318840580475, "min": 0.7362318840580475}, "precision_at_target_recall": {"count": 1, "max": 0.7994269340974787, "sum": 0.7994269340974787, "min": 0.7994269340974787}, "accuracy": {"count": 1, "max": 0.9993529423566943, "sum": 0.9993529423566943, "min": 0.9993529423566943}, "threshold_for_f1": {"count": 1, "max": 0.016658587381243706, "sum": 0.016658587381243706, "min": 0.016658587381243706}}, "EndTime": 1518539812.993149, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner"}, "StartTime": 1518539812.983724}
    [0m
    [31m#metrics {"Metrics": {"f1": {"count": 1, "max": 0.8220858895705522, "sum": 0.8220858895705522, "min": 0.8220858895705522}, "threshold_for_accuracy": {"count": 1, "max": 0.02889210544526577, "sum": 0.02889210544526577, "min": 0.02889210544526577}, "threshold_for_precision_at_target_recall": {"count": 1, "max": 0.018361520022153854, "sum": 0.018361520022153854, "min": 0.018361520022153854}, "threshold_for_recall_at_target_precision": {"count": 1, "max": 0.01731589436531067, "sum": 0.01731589436531067, "min": 0.01731589436531067}, "recall_at_precision": {"count": 1, "max": 0.8086956521739684, "sum": 0.8086956521739684, "min": 0.8086956521739684}, "precision_at_target_recall": {"count": 1, "max": 0.8195266272189883, "sum": 0.8195266272189883, "min": 0.8195266272189883}, "accuracy": {"count": 1, "max": 0.9994181497160972, "sum": 0.9994181497160972, "min": 0.9994181497160972}, "threshold_for_f1": {"count": 1, "max": 0.02889210544526577, "sum": 0.02889210544526577, "min": 0.02889210544526577}}, "EndTime": 1518539813.002439, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner"}, "StartTime": 1518539812.993238}
    [0m
    [31m#metrics {"Metrics": {"f1": {"count": 1, "max": 0.8100890207715133, "sum": 0.8100890207715133, "min": 0.8100890207715133}, "threshold_for_accuracy": {"count": 1, "max": 0.04669862613081932, "sum": 0.04669862613081932, "min": 0.04669862613081932}, "threshold_for_precision_at_target_recall": {"count": 1, "max": 0.010824650526046753, "sum": 0.010824650526046753, "min": 0.010824650526046753}, "threshold_for_recall_at_target_precision": {"count": 1, "max": 0.010824650526046753, "sum": 0.010824650526046753, "min": 0.010824650526046753}, "recall_at_precision": {"count": 1, "max": 0.8028985507246948, "sum": 0.8028985507246948, "min": 0.8028985507246948}, "precision_at_target_recall": {"count": 1, "max": 0.8099415204678918, "sum": 0.8099415204678918, "min": 0.8099415204678918}, "accuracy": {"count": 1, "max": 0.9993679902088642, "sum": 0.9993679902088642, "min": 0.9993679902088642}, "threshold_for_f1": {"count": 1, "max": 0.020745646208524704, "sum": 0.020745646208524704, "min": 0.020745646208524704}}, "EndTime": 1518539813.011702, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner"}, "StartTime": 1518539813.002522}
    [0m
    [31m#metrics {"Metrics": {"f1": {"count": 1, "max": 0.8220858895705522, "sum": 0.8220858895705522, "min": 0.8220858895705522}, "threshold_for_accuracy": {"count": 1, "max": 0.02958529442548752, "sum": 0.02958529442548752, "min": 0.02958529442548752}, "threshold_for_precision_at_target_recall": {"count": 1, "max": 0.018386855721473694, "sum": 0.018386855721473694, "min": 0.018386855721473694}, "threshold_for_recall_at_target_precision": {"count": 1, "max": 0.017356185242533684, "sum": 0.017356185242533684, "min": 0.017356185242533684}, "recall_at_precision": {"count": 1, "max": 0.8086956521739684, "sum": 0.8086956521739684, "min": 0.8086956521739684}, "precision_at_target_recall": {"count": 1, "max": 0.8195266272189883, "sum": 0.8195266272189883, "min": 0.8195266272189883}, "accuracy": {"count": 1, "max": 0.9994181497160972, "sum": 0.9994181497160972, "min": 0.9994181497160972}, "threshold_for_f1": {"count": 1, "max": 0.02958529442548752, "sum": 0.02958529442548752, "min": 0.02958529442548752}}, "EndTime": 1518539813.021184, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner"}, "StartTime": 1518539813.011789}
    [0m
    [31m#metrics {"Metrics": {"f1": {"count": 1, "max": 0.8100890207715133, "sum": 0.8100890207715133, "min": 0.8100890207715133}, "threshold_for_accuracy": {"count": 1, "max": 0.046698957681655884, "sum": 0.046698957681655884, "min": 0.046698957681655884}, "threshold_for_precision_at_target_recall": {"count": 1, "max": 0.010825625620782375, "sum": 0.010825625620782375, "min": 0.010825625620782375}, "threshold_for_recall_at_target_precision": {"count": 1, "max": 0.010825625620782375, "sum": 0.010825625620782375, "min": 0.010825625620782375}, "recall_at_precision": {"count": 1, "max": 0.8028985507246948, "sum": 0.8028985507246948, "min": 0.8028985507246948}, "precision_at_target_recall": {"count": 1, "max": 0.8099415204678918, "sum": 0.8099415204678918, "min": 0.8099415204678918}, "accuracy": {"count": 1, "max": 0.9993679902088642, "sum": 0.9993679902088642, "min": 0.9993679902088642}, "threshold_for_f1": {"count": 1, "max": 0.020745806396007538, "sum": 0.020745806396007538, "min": 0.020745806396007538}}, "EndTime": 1518539813.030645, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner"}, "StartTime": 1518539813.021298}
    [0m
    [31m#metrics {"Metrics": {"f1": {"count": 1, "max": 0.8176470588235294, "sum": 0.8176470588235294, "min": 0.8176470588235294}, "threshold_for_accuracy": {"count": 1, "max": 0.002611179370433092, "sum": 0.002611179370433092, "min": 0.002611179370433092}, "threshold_for_precision_at_target_recall": {"count": 1, "max": 0.0024112367536872625, "sum": 0.0024112367536872625, "min": 0.0024112367536872625}, "threshold_for_recall_at_target_precision": {"count": 1, "max": 0.0024112367536872625, "sum": 0.0024112367536872625, "min": 0.0024112367536872625}, "recall_at_precision": {"count": 1, "max": 0.8057971014493317, "sum": 0.8057971014493317, "min": 0.8057971014493317}, "precision_at_target_recall": {"count": 1, "max": 0.8298507462687075, "sum": 0.8298507462687075, "min": 0.8298507462687075}, "accuracy": {"count": 1, "max": 0.9993780221103108, "sum": 0.9993780221103108, "min": 0.9993780221103108}, "threshold_for_f1": {"count": 1, "max": 0.0024112367536872625, "sum": 0.0024112367536872625, "min": 0.0024112367536872625}}, "EndTime": 1518539813.040108, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner"}, "StartTime": 1518539813.030737}
    [0m
    [31m#metrics {"Metrics": {"f1": {"count": 1, "max": 0.8023088023088023, "sum": 0.8023088023088023, "min": 0.8023088023088023}, "threshold_for_accuracy": {"count": 1, "max": 0.0016964742681011558, "sum": 0.0016964742681011558, "min": 0.0016964742681011558}, "threshold_for_precision_at_target_recall": {"count": 1, "max": 0.0016964742681011558, "sum": 0.0016964742681011558, "min": 0.0016964742681011558}, "threshold_for_recall_at_target_precision": {"count": 1, "max": 0.0018729869043454528, "sum": 0.0018729869043454528, "min": 0.0018729869043454528}, "recall_at_precision": {"count": 1, "max": 0.7043478260870422, "sum": 0.7043478260870422, "min": 0.7043478260870422}, "precision_at_target_recall": {"count": 1, "max": 0.7988505747127015, "sum": 0.7988505747127015, "min": 0.7988505747127015}, "accuracy": {"count": 1, "max": 0.9993128147509079, "sum": 0.9993128147509079, "min": 0.9993128147509079}, "threshold_for_f1": {"count": 1, "max": 0.0016964742681011558, "sum": 0.0016964742681011558, "min": 0.0016964742681011558}}, "EndTime": 1518539813.049471, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner"}, "StartTime": 1518539813.040196}
    [0m
    [31m#metrics {"Metrics": {"f1": {"count": 1, "max": 0.8176470588235294, "sum": 0.8176470588235294, "min": 0.8176470588235294}, "threshold_for_accuracy": {"count": 1, "max": 0.0020974264480173588, "sum": 0.0020974264480173588, "min": 0.0020974264480173588}, "threshold_for_precision_at_target_recall": {"count": 1, "max": 0.0020974264480173588, "sum": 0.0020974264480173588, "min": 0.0020974264480173588}, "threshold_for_recall_at_target_precision": {"count": 1, "max": 0.0020974264480173588, "sum": 0.0020974264480173588, "min": 0.0020974264480173588}, "recall_at_precision": {"count": 1, "max": 0.8057971014493317, "sum": 0.8057971014493317, "min": 0.8057971014493317}, "precision_at_target_recall": {"count": 1, "max": 0.8298507462687075, "sum": 0.8298507462687075, "min": 0.8298507462687075}, "accuracy": {"count": 1, "max": 0.9993780221103108, "sum": 0.9993780221103108, "min": 0.9993780221103108}, "threshold_for_f1": {"count": 1, "max": 0.0020974264480173588, "sum": 0.0020974264480173588, "min": 0.0020974264480173588}}, "EndTime": 1518539813.058916, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner"}, "StartTime": 1518539813.049563}
    [0m
    [31m#metrics {"Metrics": {"f1": {"count": 1, "max": 0.8023088023088023, "sum": 0.8023088023088023, "min": 0.8023088023088023}, "threshold_for_accuracy": {"count": 1, "max": 0.0016964749665930867, "sum": 0.0016964749665930867, "min": 0.0016964749665930867}, "threshold_for_precision_at_target_recall": {"count": 1, "max": 0.0016964749665930867, "sum": 0.0016964749665930867, "min": 0.0016964749665930867}, "threshold_for_recall_at_target_precision": {"count": 1, "max": 0.001872989465482533, "sum": 0.001872989465482533, "min": 0.001872989465482533}, "recall_at_precision": {"count": 1, "max": 0.7043478260870422, "sum": 0.7043478260870422, "min": 0.7043478260870422}, "precision_at_target_recall": {"count": 1, "max": 0.7988505747127015, "sum": 0.7988505747127015, "min": 0.7988505747127015}, "accuracy": {"count": 1, "max": 0.9993128147509079, "sum": 0.9993128147509079, "min": 0.9993128147509079}, "threshold_for_f1": {"count": 1, "max": 0.0016964749665930867, "sum": 0.0016964749665930867, "min": 0.0016964749665930867}}, "EndTime": 1518539813.068383, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner"}, "StartTime": 1518539813.059004}
    [0m
    [31m#metrics {"Metrics": {"f1": {"count": 1, "max": 0.8176470588235294, "sum": 0.8176470588235294, "min": 0.8176470588235294}, "threshold_for_accuracy": {"count": 1, "max": 0.0020947023294866085, "sum": 0.0020947023294866085, "min": 0.0020947023294866085}, "threshold_for_precision_at_target_recall": {"count": 1, "max": 0.0020947023294866085, "sum": 0.0020947023294866085, "min": 0.0020947023294866085}, "threshold_for_recall_at_target_precision": {"count": 1, "max": 0.0020947023294866085, "sum": 0.0020947023294866085, "min": 0.0020947023294866085}, "recall_at_precision": {"count": 1, "max": 0.8057971014493317, "sum": 0.8057971014493317, "min": 0.8057971014493317}, "precision_at_target_recall": {"count": 1, "max": 0.8298507462687075, "sum": 0.8298507462687075, "min": 0.8298507462687075}, "accuracy": {"count": 1, "max": 0.9993780221103108, "sum": 0.9993780221103108, "min": 0.9993780221103108}, "threshold_for_f1": {"count": 1, "max": 0.0020947023294866085, "sum": 0.0020947023294866085, "min": 0.0020947023294866085}}, "EndTime": 1518539813.077881, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner"}, "StartTime": 1518539813.068473}
    [0m
    [31m#metrics {"Metrics": {"f1": {"count": 1, "max": 0.003455027064378671, "sum": 0.003455027064378671, "min": 0.003455027064378671}, "threshold_for_accuracy": {"count": 1, "max": Infinity, "sum": Infinity, "min": Infinity}, "threshold_for_precision_at_target_recall": {"count": 1, "max": -Infinity, "sum": -Infinity, "min": -Infinity}, "threshold_for_recall_at_target_precision": {"count": 1, "max": 0.0012821704149246216, "sum": 0.0012821704149246216, "min": 0.0012821704149246216}, "recall_at_precision": {"count": 1, "max": 0.0, "sum": 0.0, "min": 0.0}, "precision_at_target_recall": {"count": 1, "max": 0.0017305029995390332, "sum": 0.0017305029995390332, "min": 0.0017305029995390332}, "accuracy": {"count": 1, "max": 0.9982694970004614, "sum": 0.9982694970004614, "min": 0.9982694970004614}, "threshold_for_f1": {"count": 1, "max": -Infinity, "sum": -Infinity, "min": -Infinity}}, "EndTime": 1518539813.084422, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner"}, "StartTime": 1518539813.077969}
    [0m
    [31m#metrics {"Metrics": {"f1": {"count": 1, "max": 0.8176470588235294, "sum": 0.8176470588235294, "min": 0.8176470588235294}, "threshold_for_accuracy": {"count": 1, "max": 0.0021202436182647943, "sum": 0.0021202436182647943, "min": 0.0021202436182647943}, "threshold_for_precision_at_target_recall": {"count": 1, "max": 0.0020679652225226164, "sum": 0.0020679652225226164, "min": 0.0020679652225226164}, "threshold_for_recall_at_target_precision": {"count": 1, "max": 0.0020679652225226164, "sum": 0.0020679652225226164, "min": 0.0020679652225226164}, "recall_at_precision": {"count": 1, "max": 0.8057971014493317, "sum": 0.8057971014493317, "min": 0.8057971014493317}, "precision_at_target_recall": {"count": 1, "max": 0.8298507462687075, "sum": 0.8298507462687075, "min": 0.8298507462687075}, "accuracy": {"count": 1, "max": 0.9993780221103108, "sum": 0.9993780221103108, "min": 0.9993780221103108}, "threshold_for_f1": {"count": 1, "max": 0.0020679652225226164, "sum": 0.0020679652225226164, "min": 0.0020679652225226164}}, "EndTime": 1518539813.093876, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner"}, "StartTime": 1518539813.084502}
    [0m
    [31m#metrics {"Metrics": {"f1": {"count": 1, "max": 0.003455027064378671, "sum": 0.003455027064378671, "min": 0.003455027064378671}, "threshold_for_accuracy": {"count": 1, "max": Infinity, "sum": Infinity, "min": Infinity}, "threshold_for_precision_at_target_recall": {"count": 1, "max": -Infinity, "sum": -Infinity, "min": -Infinity}, "threshold_for_recall_at_target_precision": {"count": 1, "max": 0.0011913338676095009, "sum": 0.0011913338676095009, "min": 0.0011913338676095009}, "recall_at_precision": {"count": 1, "max": 0.0, "sum": 0.0, "min": 0.0}, "precision_at_target_recall": {"count": 1, "max": 0.0017305029995390332, "sum": 0.0017305029995390332, "min": 0.0017305029995390332}, "accuracy": {"count": 1, "max": 0.9982694970004614, "sum": 0.9982694970004614, "min": 0.9982694970004614}, "threshold_for_f1": {"count": 1, "max": -Infinity, "sum": -Infinity, "min": -Infinity}}, "EndTime": 1518539813.100415, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner"}, "StartTime": 1518539813.094005}
    [0m
    [31m#metrics {"Metrics": {"f1": {"count": 1, "max": 0.8100890207715133, "sum": 0.8100890207715133, "min": 0.8100890207715133}, "threshold_for_accuracy": {"count": 1, "max": 0.0019895401783287525, "sum": 0.0019895401783287525, "min": 0.0019895401783287525}, "threshold_for_precision_at_target_recall": {"count": 1, "max": 0.0019877024460583925, "sum": 0.0019877024460583925, "min": 0.0019877024460583925}, "threshold_for_recall_at_target_precision": {"count": 1, "max": 0.0019895401783287525, "sum": 0.0019895401783287525, "min": 0.0019895401783287525}, "recall_at_precision": {"count": 1, "max": 0.7913043478261474, "sum": 0.7913043478261474, "min": 0.7913043478261474}, "precision_at_target_recall": {"count": 1, "max": 0.7988505747127015, "sum": 0.7988505747127015, "min": 0.7988505747127015}, "accuracy": {"count": 1, "max": 0.9993579583074176, "sum": 0.9993579583074176, "min": 0.9993579583074176}, "threshold_for_f1": {"count": 1, "max": 0.0019895401783287525, "sum": 0.0019895401783287525, "min": 0.0019895401783287525}}, "EndTime": 1518539813.109885, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner"}, "StartTime": 1518539813.100526}
    [0m
    [31m#metrics {"Metrics": {"f1": {"count": 1, "max": 0.6228070175438597, "sum": 0.6228070175438597, "min": 0.6228070175438597}, "threshold_for_accuracy": {"count": 1, "max": 0.0015430636703968048, "sum": 0.0015430636703968048, "min": 0.0015430636703968048}, "threshold_for_precision_at_target_recall": {"count": 1, "max": 0.0015389836626127362, "sum": 0.0015389836626127362, "min": 0.0015389836626127362}, "threshold_for_recall_at_target_precision": {"count": 1, "max": 0.0015602682251483202, "sum": 0.0015602682251483202, "min": 0.0015602682251483202}, "recall_at_precision": {"count": 1, "max": 0.07246376811621087, "sum": 0.07246376811621087, "min": 0.07246376811621087}, "precision_at_target_recall": {"count": 1, "max": 0.269794721407696, "sum": 0.269794721407696, "min": 0.269794721407696}, "accuracy": {"count": 1, "max": 0.9987058847133886, "sum": 0.9987058847133886, "min": 0.9987058847133886}, "threshold_for_f1": {"count": 1, "max": 0.0015430636703968048, "sum": 0.0015430636703968048, "min": 0.0015430636703968048}}, "EndTime": 1518539813.119078, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner"}, "StartTime": 1518539813.110016}
    [0m
    [31m#metrics {"Metrics": {"f1": {"count": 1, "max": 0.260343087790111, "sum": 0.260343087790111, "min": 0.260343087790111}, "threshold_for_accuracy": {"count": 1, "max": Infinity, "sum": Infinity, "min": Infinity}, "threshold_for_precision_at_target_recall": {"count": 1, "max": 0.0017423316603526473, "sum": 0.0017423316603526473, "min": 0.0017423316603526473}, "threshold_for_recall_at_target_precision": {"count": 1, "max": 0.0017408719286322594, "sum": 0.0017408719286322594, "min": 0.0017408719286322594}, "recall_at_precision": {"count": 1, "max": 0.0, "sum": 0.0, "min": 0.0}, "precision_at_target_recall": {"count": 1, "max": 0.01956368754399001, "sum": 0.01956368754399001, "min": 0.01956368754399001}, "accuracy": {"count": 1, "max": 0.9982694970004614, "sum": 0.9982694970004614, "min": 0.9982694970004614}, "threshold_for_f1": {"count": 1, "max": 0.0017436194466426969, "sum": 0.0017436194466426969, "min": 0.0017436194466426969}}, "EndTime": 1518539813.126206, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner"}, "StartTime": 1518539813.119169}
    [0m
    [31m#metrics {"Metrics": {"f1": {"count": 1, "max": 0.622478386167147, "sum": 0.622478386167147, "min": 0.622478386167147}, "threshold_for_accuracy": {"count": 1, "max": 0.0015434155939146876, "sum": 0.0015434155939146876, "min": 0.0015434155939146876}, "threshold_for_precision_at_target_recall": {"count": 1, "max": 0.0015389265026897192, "sum": 0.0015389265026897192, "min": 0.0015389265026897192}, "threshold_for_recall_at_target_precision": {"count": 1, "max": 0.0015602532075718045, "sum": 0.0015602532075718045, "min": 0.0015602532075718045}, "recall_at_precision": {"count": 1, "max": 0.07246376811621087, "sum": 0.07246376811621087, "min": 0.07246376811621087}, "precision_at_target_recall": {"count": 1, "max": 0.2598870056497872, "sum": 0.2598870056497872, "min": 0.2598870056497872}, "accuracy": {"count": 1, "max": 0.9987259485162818, "sum": 0.9987259485162818, "min": 0.9987259485162818}, "threshold_for_f1": {"count": 1, "max": 0.0015428601764142513, "sum": 0.0015428601764142513, "min": 0.0015428601764142513}}, "EndTime": 1518539813.135391, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner"}, "StartTime": 1518539813.126293}
    [0m
    [31m#metrics {"Metrics": {"f1": {"count": 1, "max": 0.8023088023088023, "sum": 0.8023088023088023, "min": 0.8023088023088023}, "threshold_for_accuracy": {"count": 1, "max": 0.0017262108158320189, "sum": 0.0017262108158320189, "min": 0.0017262108158320189}, "threshold_for_precision_at_target_recall": {"count": 1, "max": 0.0017262108158320189, "sum": 0.0017262108158320189, "min": 0.0017262108158320189}, "threshold_for_recall_at_target_precision": {"count": 1, "max": 0.001729545183479786, "sum": 0.001729545183479786, "min": 0.001729545183479786}, "recall_at_precision": {"count": 1, "max": 0.6579710144928528, "sum": 0.6579710144928528, "min": 0.6579710144928528}, "precision_at_target_recall": {"count": 1, "max": 0.7988505747127015, "sum": 0.7988505747127015, "min": 0.7988505747127015}, "accuracy": {"count": 1, "max": 0.9993128147509079, "sum": 0.9993128147509079, "min": 0.9993128147509079}, "threshold_for_f1": {"count": 1, "max": 0.0017262108158320189, "sum": 0.0017262108158320189, "min": 0.0017262108158320189}}, "EndTime": 1518539813.144822, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner"}, "StartTime": 1518539813.135483}
    [0m
    [31m#metrics {"Metrics": {"f1": {"count": 1, "max": 0.008113590263691683, "sum": 0.008113590263691683, "min": 0.008113590263691683}, "threshold_for_accuracy": {"count": 1, "max": Infinity, "sum": Infinity, "min": Infinity}, "threshold_for_precision_at_target_recall": {"count": 1, "max": 0.0012073565740138292, "sum": 0.0012073565740138292, "min": 0.0012073565740138292}, "threshold_for_recall_at_target_precision": {"count": 1, "max": 0.0012612584978342056, "sum": 0.0012612584978342056, "min": 0.0012612584978342056}, "recall_at_precision": {"count": 1, "max": 0.0, "sum": 0.0, "min": 0.0}, "precision_at_target_recall": {"count": 1, "max": 0.0017308329068967412, "sum": 0.0017308329068967412, "min": 0.0017308329068967412}, "accuracy": {"count": 1, "max": 0.9982694970004614, "sum": 0.9982694970004614, "min": 0.9982694970004614}, "threshold_for_f1": {"count": 1, "max": 0.0012906865449622273, "sum": 0.0012906865449622273, "min": 0.0012906865449622273}}, "EndTime": 1518539813.151292, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner"}, "StartTime": 1518539813.144912}
    [0m
    [31m#metrics {"Metrics": {"f1": {"count": 1, "max": 0.2680628272251309, "sum": 0.2680628272251309, "min": 0.2680628272251309}, "threshold_for_accuracy": {"count": 1, "max": Infinity, "sum": Infinity, "min": Infinity}, "threshold_for_precision_at_target_recall": {"count": 1, "max": 0.0017176861874759197, "sum": 0.0017176861874759197, "min": 0.0017176861874759197}, "threshold_for_recall_at_target_precision": {"count": 1, "max": 0.0017167814075946808, "sum": 0.0017167814075946808, "min": 0.0017167814075946808}, "recall_at_precision": {"count": 1, "max": 0.0, "sum": 0.0, "min": 0.0}, "precision_at_target_recall": {"count": 1, "max": 0.02000000000000705, "sum": 0.02000000000000705, "min": 0.02000000000000705}, "accuracy": {"count": 1, "max": 0.9982694970004614, "sum": 0.9982694970004614, "min": 0.9982694970004614}, "threshold_for_f1": {"count": 1, "max": 0.0017188951605930924, "sum": 0.0017188951605930924, "min": 0.0017188951605930924}}, "EndTime": 1518539813.15843, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner"}, "StartTime": 1518539813.151376}
    [0m
    [31m#metrics {"Metrics": {"f1": {"count": 1, "max": 0.008113590263691683, "sum": 0.008113590263691683, "min": 0.008113590263691683}, "threshold_for_accuracy": {"count": 1, "max": Infinity, "sum": Infinity, "min": Infinity}, "threshold_for_precision_at_target_recall": {"count": 1, "max": 0.001207358785904944, "sum": 0.001207358785904944, "min": 0.001207358785904944}, "threshold_for_recall_at_target_precision": {"count": 1, "max": 0.0012574447318911552, "sum": 0.0012574447318911552, "min": 0.0012574447318911552}, "recall_at_precision": {"count": 1, "max": 0.0, "sum": 0.0, "min": 0.0}, "precision_at_target_recall": {"count": 1, "max": 0.0017308329068967412, "sum": 0.0017308329068967412, "min": 0.0017308329068967412}, "accuracy": {"count": 1, "max": 0.9982694970004614, "sum": 0.9982694970004614, "min": 0.9982694970004614}, "threshold_for_f1": {"count": 1, "max": 0.0012906889896839857, "sum": 0.0012906889896839857, "min": 0.0012906889896839857}}, "EndTime": 1518539813.16496, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner"}, "StartTime": 1518539813.158511}
    [0m
    [31m[02/13/2018 16:36:53 INFO 140471961589568] Selection criteria: accuracy[0m
    [31mmodel: 12[0m
    [31mthreshold: 0.028892[0m
    [31mscore: 0.999418[0m
    [31m[02/13/2018 16:36:53 INFO 140471961589568] Saved checkpoint to "/tmp/tmpEOoHMS/mx-mod-0000.params"[0m
    [31m#metrics {"Metrics": {"totaltime": {"count": 1, "max": 415222.3379611969, "sum": 415222.3379611969, "min": 415222.3379611969}, "finalize.time": {"count": 1, "max": 7318.602085113525, "sum": 7318.602085113525, "min": 7318.602085113525}, "initialize.time": {"count": 1, "max": 3165.503978729248, "sum": 3165.503978729248, "min": 3165.503978729248}, "setuptime": {"count": 1, "max": 31.399011611938477, "sum": 31.399011611938477, "min": 31.399011611938477}, "update.time": {"count": 10, "max": 41599.754095077515, "sum": 404643.2693004608, "min": 40138.32497596741}, "epochs": {"count": 1, "max": 10, "sum": 10.0, "min": 10}}, "EndTime": 1518539813.180422, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner"}, "StartTime": 1518539398.044161}
    [0m
    ===== Job Complete =====


### Monitoring your Training Job
Amazon SageMaker has logs training using Amazon CloudWatch. The logs are located under aws/TrainingJobs
![training logs](../images/logging.png)

Amazon SageMaker also provide a set of Metrics. Metrics include:
- Memory Utilization
- CPU Utilization
- GPU Unitlization
- Disk Utilization
You can use these metrics to size your training cluster accordingly.
![training logs](../images/metrics.png)

We would also measure whether we have chosen the correct EC2 profile for the model training. The below screenshot shows utilization of the resources.

![util](../images/underutilization.png)

We can generally observe that the p2.xlarge is an overkill.

## SageMaker Process - Hosting the model
Now that we have a trained and saved model, we can venture into creating endpoints. Amazon SageMaker requires a single line of code in order to created a fully managed and elasticlly scalable endpoint environemnt. 

Amazon SageMaker takes your model file from the S3 bucket and dockerizes it. It also creates an Amazon [ECS (Amazon Elastic Container Service)](https://aws.amazon.com/documentation/ecs/) infrastructre, fronted with an [ELB or ElasticLoadBalncer](https://aws.amazon.com/documentation/elastic-load-balancing/).

The instances are also members of an [Autoscaling Group](https://docs.aws.amazon.com/autoscaling/ec2/userguide/AutoScalingGroup.html), meaning that based on the incoming load the number of docker images and underlying EC2 instances can growand shrink dynamically. 

Model hosting and Deployment is independent of how you develop your models, meaning that you do not need to develop your model within Amazon SageMaker or use Amazon Algorithms in order to host your models in Amazon SageMaker. 

This post is foused on LinearLearning and hosting your models is out of the scope of this blog post. For more information please refer to [Amazon SageMaker doumentation](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms.html)

From the class diaram we delved into previously, we can remember that EstimatorBase implements a method called `deploy()`. All the deployment magic happens through a single call to `Estimator.deploy()`

We pass initial cluster size and instance type to `deploy()`. 
```python
linear_predictor = linear.deploy(initial_instance_count=1, #Initial number of instances. 
                                                           #Autoscaling can increase the number of instances.
                                 instance_type='ml.m4.xlarge') # instance type
```

Executing `deploy()` calls `create_model()`. `create_model()` is an abstract method of EstimatorBase and is implemented by its sub-classes. `create_model()` in turn returns a Model object. calling  `deploy()` returns a RealTimePredictor class, that can be in fact deployed to the live environement.

### Model Class Hierarchy
![Model](../images/model.png)


### Predictor Class 
![predictor](../images/predictor.png)



```python
linear_predictor = linear.deploy(initial_instance_count=1, #Initial number of instances. 
                                                           #Autoscaling can increase the number of instances.
                                 instance_type='ml.m4.xlarge') # instance type
```

    INFO:sagemaker:Creating model with name: linear-learner-2018-02-13-15-19-04-112
    INFO:sagemaker:Creating endpoint with name linear-learner-2018-02-13-13-48-23-462


    ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------!


```python
type(linear_predictor)
```




    sagemaker.predictor.RealTimePredictor



## SageMaker Process - Prediction
We the print in the previous snippet we can see that `Estimator.deploy()` eventually returns a `sagemaker.predictor.RealTimePredictor` object. `sagemaker.predictor.RealTimePredictor` implements a method called `predict()`, which is used to make live predictions.

Predictors in sagemaker accept csv and json. In this case we use json serialization.


```python
linear_predictor.content_type = 'text/csv'
linear_predictor.serializer = csv_serializer
linear_predictor.deserializer = json_deserializer
```

Since Fraudulant records are rate, I have created an array of all records in validation set with *"Class==1"* in order to test model accuracy by testing the predictor endpoint on ranges that in fact includes fraudulant transactions.

You can observe that record number 516 is correctly predicted to be a fraud when we would run prediction in range of 515-519

*
{'predictions': [{'score': 0.0006907652714289725, 'predicted_label': 0.0}, **{'score': 0.9957004189491272, 'predicted_label': 1.0}**, {'score': 0.0006745134014636278, 'predicted_label': 0.0}, {'score': 0.0006603851797990501, 'predicted_label': 0.0}]}
*



```python
#since score==1 is very rare we want to make sure we can correctly predict fradulant transaction. 
#First we print a lost of all labels where score == 1, then then run a prediction
t = []
for i in range(len(train_label)):
    if train_label[i] == 1:
        t.append(i)
        
print(t)

print('\n')
print(linear_predictor.predict(train_set[0][515:519]))

```

    [516, 765, 867, 2636, 2704, 3360, 3583, 4111, 4482, 5115, 6463, 7810, 8008, 8420, 9171, 9178, 10236, 10591, 10604, 10701, 11363, 13516, 14479, 15167, 15396, 15949, 16334, 16348, 16982, 17309, 17698, 18155, 19023, 19048, 19116, 20236, 20382, 20701, 21038, 21080, 21509, 22228, 24887, 25425, 25840, 26055, 26797, 27138, 27982, 28853, 30106, 30733, 31027, 31317, 31353, 32757, 33018, 33285, 33710, 34422, 34926, 35447, 35595, 35699, 35700, 35922, 37207, 39095, 39165, 39623, 39954, 40098, 40486, 41384, 41871, 42159, 43274, 43573, 43641, 43798, 43949, 44281, 44954, 45959, 46102, 48351, 48763, 49023, 49041, 49576, 52158, 52165, 53136, 54002, 54994, 57241, 57242, 58254, 58297, 58298, 58707, 59283, 59350, 59791, 59795, 59966, 59981, 60137, 60226, 60260, 60705, 62107, 62426, 64015, 64309, 64645, 65028, 66338, 68281, 69096, 69159, 69280, 69344, 69487, 69530, 71003, 71055, 71580, 72302, 73057, 73358, 73389, 73644, 75015, 75384, 77773, 77821, 77843, 78179, 79448, 79479, 80184, 82184, 82484, 83329, 85205, 85541, 85631, 85798, 86175, 86542, 86984, 87085, 87275, 87322, 90181, 90636, 91183, 94353, 94395, 94646, 94958, 95182, 96212, 97344, 98572, 98904, 99231, 99719, 99722, 99935, 100199, 100509, 101493, 101808, 102346, 102709, 103100, 103570, 103714, 104054, 104246, 104450, 105055, 105240, 106181, 106214, 106812, 106911, 107044, 108243, 108951, 109621, 111525, 111778, 112474, 112756, 113534, 113714, 114785, 115154, 115852, 117623, 117682, 118246, 118937, 119090, 119616, 119929, 121209, 121536, 121771, 122555, 123096, 123482, 123501, 124644, 124673, 126110, 126147, 126656, 126959, 128237, 128689, 129194, 131229, 131504, 131520, 131533, 131623, 131931, 132397, 132414, 132459, 132480, 132857, 132932, 133089, 134211, 134907, 135081, 135946, 136203, 136905, 137200, 137212, 137712, 140411, 140648, 142014, 142217, 142953, 143797, 143830, 144571, 145568, 145766, 146034, 146213, 147399, 147729, 148246, 149001, 149283, 149550, 149921, 149951, 150637, 151396, 151600, 151949, 152459, 152720, 152962, 153407, 153879, 154322, 154790, 156451, 156862, 157117, 157187, 158781, 159248, 160650, 161017, 161281, 161672, 164168, 164349, 165301, 166123, 166517, 166712, 167151, 168004, 168814, 168852, 169054, 169340, 169606, 169662, 170038, 170170, 170371, 171193, 171718, 173451, 174960, 175122, 175149, 175234, 177779, 178221, 179529, 180132, 180615, 181925, 183154, 183253, 184306, 186262, 186445, 186468, 187248, 187564, 188202, 188639, 189316, 190103, 191011, 191782, 192915, 193004, 193051, 193233, 193468, 194871, 195540, 196806, 197501, 197669, 198918, 199113, 199305]
    
    
    {'predictions': [{'score': 0.00013768798089586198, 'predicted_label': 0.0}, {'score': 1.0, 'predicted_label': 1.0}, {'score': 0.0002609600778669119, 'predicted_label': 0.0}, {'score': 0.00021634736913256347, 'predicted_label': 0.0}]}



```python
non_zero = np.count_nonzero(test_set[1])
zero = len(test_set[1]) - non_zero
print("validation set includes: {} non zero and {} items woth value zero".format(non_zero, zero))
```

    validation set includes: 147 non zero and 85296 items woth value zero



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
      <td>85265</td>
      <td>31</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>29</td>
      <td>118</td>
    </tr>
  </tbody>
</table>
</div>



### Analyzing the Results
The confusion matrix above indicates that:
- Total fraudulent transactions: 147
 - Num Examples (NE) = 85443
 - True Positive (TP) = 118
 - False Positive (FP) = 31
 - False Negative (FN) = 29

- **Recall** = TP/(TP+FN) = 118/(118+29) = 0.80
- **Precision** = TP/(TP+FP) = 118/(118+31) = 0.79
- **Accuracy** = 1- (FP+FN)/NE = 1 - (60/85443) = 0.99

The model produces a superb accuracy, but how about precision and recall? Can we improve recall to be able to pick out more fraudulent transactions accurately?

We can optimize the model to balance between recall and precision. In part 3, Excel in tuning models using Amazon LinearLearner Algorithm, I will attempt to tune the model to prioritize recall over accuracy and report the results. The dataset has low dimensions and we are using HPO, so we might not be able to improve the results in any meaningful way. It is worth experimenting though.

The model performance is based on default parameters that are located in training instances in `/opt/ml/input/config/hyperparameters.json`

The default parameters are:

u'epochs': u'10', u'init_bias': u'0.0', u'lr_scheduler_factor': u'0.99', u'num_calibration_samples': u'10000000', u'_num_kv_servers': u'auto', u'use_bias': u'true', u'num_point_for_scaler': u'10000', u'_log_level': u'info', u'bias_lr_mult': u'10', u'lr_scheduler_step': u'100', u'init_method': u'uniform', u'init_sigma': u'0.01', u'lr_scheduler_minimum_lr': u'0.00001', **u'target_recall': u'0.8'**, **u'num_models': u'32'**, u'momentum': u'0.0', u'unbias_label': u'auto', u'wd': u'0.0', u'optimizer': u'adam', u'learning_rate': u'auto', u'_kvstore': u'auto', **u'normalize_data': u'true'**, **u'binary_classifier_model_selection_criteria': u'accuracy'**, u'use_lr_scheduler': u'true', **u'target_precision': u'0.8'**, u'force_dense': u'true', u'unbias_data': u'auto', u'init_scale': u'0.07', u'bias_wd_mult': u'0', u'mini_batch_size': u'1000', u'beta_1': u'0.9', u'loss': u'auto', u'beta_2': u'0.999', u'normalize_label': u'auto', u'_num_gpus': u'auto', u'_data_format': u'record', u'positive_example_weight_mult': u'1.0', u'l1': u'0.0'}le_weight_mult': u'1.0', u'l1': u'0.0'}



A detailed description of the parameters can be found [in the Amazon SageMaker documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/ll_hyperparameters.html)

The most likely parameters that could be useful for improving recall are:
- binary_classifier_model_selection_criteria
- target_recall: *If selection criterial is set to prioritized for recall, then this value is ignored*
- target_precision
- num_models


### Monitor the live environment
A key feature of monitoring SageMaker Endpoints is that you do not monitor infrastructure, instead, you are monitoring your models even if they might be deployed on multiple machines. This gives you a holistic view of your model performance.

Amazon Cloud Watch Logs includes /aws/sagemaker/Endpoints per endpoint you create. You can use logs to monitor your model logs including all the calls to your endpoints.

![endpoint_logs](../images/logging.png)
For a graphical view of your metrics you can use Amazon CloutWatch Metrics. You can monitor your model metrics such as:
- GPU Utilization
- CPU Utilization
- Latency
- Memory Utilization

for more infomation please refer to [SageMaker documentation](!https://docs.aws.amazon.com/sagemaker/latest/dg/monitoring-overview.html)

the following screenshot is the results of several thousands of endpoint calls I performed on my linear learner.

![monitoring](../images/metrics.png)

# (optional) Delete the endpoint
f you're ready to be done with this notebook, please run the delete_endpoint line in the cell below. This will remove the hosted endpoint you created and avoid any charges from a stray instance being left on.


```python
linear.delete_endpoint()
```

# End of Part 2
PREVIOUS: [Part1: Linear Regression and Binary Classification, a Friendly Introduction](linearlearner-blogpost-part1.ipynb)

NEXT: [Part3: Excel in tuning models using Amazon LinearLearner Algorithm](linearlearner-blogpost-part3.ipynb)
