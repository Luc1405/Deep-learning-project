# Deep Learning Project
Project voor het vak Deep Learning en Dynamical Systems van de minor Data Science.

Groep bestaat uit:
  - Dominique Kuijten
  - Jarell Wespel
  - Maurits Hanhart
  - Luc Karlas


# Message Spam detecting

In today's world there is more and more need of a spam detector, this is of course because the increase of spam messages is increasing and getting more difficult to separate from real messages. In this project, we propose to use neural network and natural language processing to classify messages as either spam or as legitimate. The dataset that is used is sourced from kaggle by Team Ai https://www.kaggle.com/datasets/team-ai/spam-text-message-classification/data, this dataset contains a large number of messages with their corresponding classification. The labels consist of Spam for messages that are classified spam messages and Ham for messages that are legitimate. 


# Data cleaning 

Firstly changes were made to change the training data is the labelling. This is by changing spam to 1 and ham replaced by 0.

# Usage
To train the model, install all the necessary packages. After that, you can run the python file to train the model and use your own input to test the model. Longer messages are preferred, because the model does not handle short messages all that well.
