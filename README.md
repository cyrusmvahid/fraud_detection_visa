# Using Amazon Linear Learner and Amazon SageMaker to create realtime fraud detection endpoints

Surprisingly, many problems in the real-world can be approximated to a linear problem, meaning that a linear polynomial can provide a good output approximation to an n-dimensional input vector. Linear regression is perhaps the most applied and simplest of all machine learning models.

With this in mind, we have implemented a scalable linear regression model as a part of Amazon Algorithms that can be used in Amazon SageMaker.

In [first part](/src/linearlearner-blogpost-part1.ipynb) of this post I intend to provide an easy and intuitive introduction to linear regression as well as providing references to implement your own linear regression, both from scratch and using MXNet and Gluon.

In [part 2, Getting Hands-On with Linear Learner and Amazon SageMaker](/src/linearlearner-blogpost-part2.ipynb), I use the Visa Credit Card Fraud dataset from Kaggle, pre-process the data, and use Amazon LinearLearner to predict fraudulent transactions. During the course of Part 2 of the blog I walk through an entire processes of data pre-processing, training, and deployment of a model as a live endpoint.

In part 2 of the blog, we observe that using default values of LinearLeaner yields an impressive precision of near certainty. However, the recall on the fraudulent transactions is 80%. 

In part 3 of the blog, [Excel in tuning models using Amazon LinearLearner Algorithm](/src/linearlearner-blogpost-part3.ipynb), I attempt to fine-tune the model on the Visa dataset to see whether or not the recall could be improved.
