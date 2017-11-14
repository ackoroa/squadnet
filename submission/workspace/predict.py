print "Imports and Settings"
import csv, json, string, re, time

import numpy as np

from nltk import word_tokenize

from squadnet import *
from squadsettings import *


print "Read Word Vectors"
glove = {}
f = open('data/glove.6B.' + str(WORD_VECTOR_SIZE) + 'd.txt', 'rb')
reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
for row in reader:
    key = row[0]
    vector = map(float, row[1:])
    glove[key] = np.array(vector, dtype=np.float32).reshape(1,-1)
len(glove)


print "Read Dataset"
def text2vec(text):
    tokens = word_tokenize(text.lower())
    textVec = np.array([])
    for tok in tokens:
        textVec = np.append(textVec, glove.get(tok, np.zeros((1,WORD_VECTOR_SIZE), dtype=np.float32)))
    return textVec.reshape(1, -1)

test = []
for jsonRow in json.loads(open('data/test.json', 'rb').read()):
    for paragraph in jsonRow['paragraphs']:
        ctx = paragraph['context']
        ctxVec = text2vec(paragraph['context'])
        
        for qnaJson in paragraph['qas']:
            qnId = qnaJson['id']
            qnVec = text2vec(qnaJson['question'])            
            test.append((ctxVec, qnVec, qnId, ctx))


print "Create Model"
model = SquadNet(WORD_VECTOR_SIZE, H_SIZE, POOL_SIZE, DROPOUT_RATE, USE_GPU)
if USE_GPU:
    model.to_gpu()
  

print "Output Answers"
def get_test_batch(i, batch_size, data):
    j = min(i + batch_size, len(data))
    
    ctx = []
    qn = []
    ids = []
    ctxStrs = []
    
    cmax = 0
    qmax = 0
    for k in range(i, j):
        c, q, s, e = data[k]
        ctx.append(c)
        qn.append(q)
        ids.append(s)
        ctxStrs.append(e)
        
        cmax = max(cmax, c.shape[1])
        qmax = max(qmax, q.shape[1])

    cVec = np.zeros((len(ctx), cmax), dtype=np.float32)
    qVec = np.zeros((len(ctx), qmax), dtype=np.float32)        
    for i in range(len(ctx)):
        cVec[i, 0:ctx[i].shape[1]] = ctx[i]
        qVec[i, 0:qn[i].shape[1]] = qn[i]
    
    return Variable(cVec),            Variable(qVec),            ids,            ctxStrs

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def test_model(test_batch_size, test_print_interval, model_file,out_file):    
    serializers.load_npz(model_file, model)

    f = open(out_file, 'wb')
    out = csv.writer(f)
    out.writerow(["Id", "Answer"])

    startTime = time.time()

    with chainer.using_config('train', False):
        for i in range(0, len(test), test_batch_size):
            ctx, qn, qnId, ctxStr = get_test_batch(i, test_batch_size, test)
            model.reset_state()
            start, end = model(ctx, qn)

            for j in range(len(qnId)):
                contextTokens = word_tokenize(ctxStr[j])

                s = F.argmax(start[j]).data
                e = F.argmax(end[j]).data

                s = min(s, len(contextTokens)-1)
                e = max(e, s)
                e = min(e, len(contextTokens)-1)        

                ans = ""
                for k in range(s, e + 1):
                    ans += contextTokens[k] + " "

                out.writerow([qnId[j], normalize_answer(ans).encode('utf-8')])

            if i % test_print_interval == 0:
                print i, "/", len(test), "(" + str(time.time() - startTime) + "s)"
                startTime = time.time()

    f.close()


test_model(MINI_BATCH_SIZE, 1000, 'squadnet.model','test.csv')
