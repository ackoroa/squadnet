# Download GloVe Word Vectors
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
rm glove.6B.zip 

# Download train and test data
wget https://www.kaggle.com/c/cs5242-project-2/download/train.json
wget https://www.kaggle.com/c/cs5242-project-2/download/test.json

# Install environment
pip install -r requirements.txt

# Download nltk model for word tokenizer
python -m nltk.downloader all

#Train Model
python train.py

#Test Model
python test.py
