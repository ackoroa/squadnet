echo "Download GloVe Word Vectors"
mkdir data
wget http://nlp.stanford.edu/data/glove.6B.zip -O data/glove.6B.zip
unzip data/glove.6B.zip -d data/
rm data/glove.6B.zip 

echo "Install environment"
pip install -r requirements.txt

echo "Download nltk model for word tokenizer"
python -m nltk.downloader all

echo "Train Model"
python train.py

echo "Test Model"
python predict.py
