#Stack Exchange Recommender

A recommender for questions posted on Stack Exchange based on a user's favorite questions. The system returns a set of recommended questions given a stack exchange user ID. System is built on top of PyTorch and Python3.

##Model

Recommender is structured as a classification problem. A four-layer fully connected neural network is used to produce recommendation probabilities of questions stored in the system. 

Pairs of user-question embeddings are fed into the classifier as inputs, which are separately trained using the Skip-gram model. User embeddings are trained based on the assumption that users who favorite the same question are similar. Question embeddings are trained similarly, using groups of related questions.

Data used is collected via the Stack Exchange public API. Current model is trained on 150 users and 26,000 questions. System allows more data to be collected.

##Usage

* Download the project, data already collected are stored in a SQLite database (storage/cla_021.db)
* Acquire Stack Exchange API Key on: https://api.stackexchange.com/
* Create file with path "resource/key", store API key in the file
* Call \_init\_user.py, pass user ID as parameter
* The system will retrain user embeddings and return recommendations

```
$ python _init_user.py [user_id]
```
##Retrain Question Embeddings

Question embeddings can be retrained on new data to get recommendations on recently posted questions.

```
$ python _init_retrain.py [startdate] [enddate]
#date format: yyyy-mm-dd
```
