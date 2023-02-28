https://github.com/lazyprogrammer/machine_learning_examples





machine_learning_examples
=========================

A collection of machine learning examples and tutorials.

Find associated tutorials at https://lazyprogrammer.me

Find associated courses at https://deeplearningcourses.com

Please note that not all code from all courses will be found in this repository. Some newer code examples (e.g. most of Tensorflow 2.0) were done in Google Colab. Therefore, you should check the instructions given in the lectures for the course you are taking.


How to I find the code for a particular course?
===============================================

The code for each course is separated by folder. You can determine which folder corresponds with which course by watching the "Where to get the code" lecture inside the course (usually Lecture 2 or 3).

Remember: one folder = one course.


Why you should not fork this repo
=================================

I've noticed that many people have out-of-date forks. Thus, I recommend not forking this repository if you take one of my courses. I am constantly updating my courses, and your fork will soon become out-of-date. You should clone the repository instead to make it easy to get updates (i.e. just "git pull" randomly and frequently).


Where is the code for your latest courses?
==========================================

Beginning with Tensorflow 2, I started to use Google Colab. For those courses, unless otherwise noted, the code will be on Google Colab. Links to the notebooks are provided in the course. See the lecture "Where to get the code" for further details.


VIP Course Links
===================

**Data Science: Transformers for Natural Language Processing**

https://deeplearningcourses.com/c/data-science-transformers-nlp


**Machine Learning: Natural Language Processing in Python (V2)**

https://deeplearningcourses.com/c/natural-language-processing-in-python


**Time Series Analysis, Forecasting, and Machine Learning**

https://deeplearningcourses.com/c/time-series-analysis


**Financial Engineering and Artificial Intelligence in Python**

https://deeplearningcourses.com/c/ai-finance


**PyTorch: Deep Learning and Artificial Intelligence**

https://deeplearningcourses.com/c/pytorch-deep-learning


**Tensorflow 2.0: Deep Learning and Artificial Intelligence** (VIP Version)

https://deeplearningcourses.com/c/deep-learning-tensorflow-2



Deep Learning Courses Exclusives
================================

Data Science: Bayesian Linear Regression in Python
https://deeplearningcourses.com/c/bayesian-linear-regression-in-python

Data Science: Bayesian Classification in Python
https://deeplearningcourses.com/c/bayesian-classification-in-python

Classical Statistical Inference and A/B Testing in Python
https://deeplearningcourses.com/c/statistical-inference-in-python

Linear Programming for Linear Regression in Python
https://deeplearningcourses.com/c/linear-programming-python

MATLAB for Students, Engineers, and Professionals in STEM
https://deeplearningcourses.com/c/matlab



Other Course Links
==================

Math 0-1: Calculus for Data Science & Machine Learning
https://deeplearningcourses.com/c/calculus-data-science

Data Science & Machine Learning: Naive Bayes in Python
https://deeplearningcourses.com/c/data-science-machine-learning-naive-bayes-in-python

Cutting-Edge AI: Deep Reinforcement Learning in Python
https://deeplearningcourses.com/c/cutting-edge-artificial-intelligence

Recommender Systems and Deep Learning in Python
https://deeplearningcourses.com/c/recommender-systems

Machine Learning and AI: Support Vector Machines in Python
https://deeplearningcourses.com/c/support-vector-machines-in-python

Deep Learning: Advanced Computer Vision
https://deeplearningcourses.com/c/advanced-computer-vision

Deep Learning: Advanced NLP and RNNs
https://deeplearningcourses.com/c/deep-learning-advanced-nlp

Deep Learning: GANs and Variational Autoencoders
https://deeplearningcourses.com/c/deep-learning-gans-and-variational-autoencoders

Advanced AI: Deep Reinforcement Learning in Python
https://deeplearningcourses.com/c/deep-reinforcement-learning-in-python

Artificial Intelligence: Reinforcement Learning in Python
https://deeplearningcourses.com/c/artificial-intelligence-reinforcement-learning-in-python

Natural Language Processing with Deep Learning in Python
https://deeplearningcourses.com/c/natural-language-processing-with-deep-learning-in-python

Deep Learning: Recurrent Neural Networks in Python
https://deeplearningcourses.com/c/deep-learning-recurrent-neural-networks-in-python

Unsupervised Machine Learning: Hidden Markov Models in Python
https://deeplearningcourses.com/c/unsupervised-machine-learning-hidden-markov-models-in-python

Deep Learning Prerequisites: The Numpy Stack in Python
https://deeplearningcourses.com/c/deep-learning-prerequisites-the-numpy-stack-in-python

Deep Learning Prerequisites: Linear Regression in Python
https://deeplearningcourses.com/c/data-science-linear-regression-in-python

Deep Learning Prerequisites: Logistic Regression in Python
https://deeplearningcourses.com/c/data-science-logistic-regression-in-python

Data Science: Deep Learning and Neural Networks in Python
https://deeplearningcourses.com/c/data-science-deep-learning-in-python

Cluster Analysis and Unsupervised Machine Learning in Python
https://deeplearningcourses.com/c/cluster-analysis-unsupervised-machine-learning-python

Data Science: Supervised Machine Learning in Python
https://deeplearningcourses.com/c/data-science-supervised-machine-learning-in-python

Bayesian Machine Learning in Python: A/B Testing
https://deeplearningcourses.com/c/bayesian-machine-learning-in-python-ab-testing

Data Science: Natural Language Processing in Python
https://deeplearningcourses.com/c/data-science-natural-language-processing-in-python

Modern Deep Learning in Python
https://deeplearningcourses.com/c/data-science-deep-learning-in-theano-tensorflow

Ensemble Machine Learning in Python: Random Forest and AdaBoost
https://deeplearningcourses.com/c/machine-learning-in-python-random-forest-adaboost

Deep Learning: Convolutional Neural Networks in Python
https://deeplearningcourses.com/c/deep-learning-convolutional-neural-networks-theano-tensorflow

Unsupervised Deep Learning in Python
https://deeplearningcourses.com/c/unsupervised-deep-learning-in-python












--------------------------------------------------------DATA LINK-----------------------------------------------------
Colab Basics
TF2.0 Loading Data
# download the data from a URL
# source: https://archive.ics.uci.edu/ml/datasets/Arrhythmia
# alternate URL: https://lazyprogrammer.me/course_files/arrhythmia.data
#!wget --no-check-certificate https://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.data
!wget https://lazyprogrammer.me/course_files/arrhythmia.data


# use keras get_file to download the auto MPG dataset
# source: https://archive.ics.uci.edu/ml/datasets/Auto+MPG
#url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'


### alternate URL
url = 'https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/tf2.0/auto-mpg.data'

# another method: upload your own file


##### PLEASE NOTE: IT DOES NOT MATTER WHICH FILE YOU UPLOAD
##### YOU CAN UPLOAD ANY FILE YOU WANT
##### IN FACT, YOU ARE ENCOURAGED TO EXPLORE ON YOUR OWN


# if you must, then get the file from here:
# https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/tf2.0/daily-minimum-temperatures-in-me.csv

Machine Learning Basics
TF2.0 Linear Regression
!wget https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/tf2.0/moore.csv

RNN
TF2.0 Stock Returns
# yes, you can read dataframes from URLs!
df = pd.read_csv('https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/tf2.0/sbux.csv')

Natural Language Processing
TF2.0 Spam Detection
# Unfortunately this URL doesn't work directly with pd.read_csv
!wget -nc https://lazyprogrammer.me/course_files/spam.csv





