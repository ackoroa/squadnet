# # Imports and Settings
import csv, json, string, re, time
from random import shuffle

import numpy as np

from nltk import word_tokenize

from squadnet import *
from squadsettings import *


# # Read Word Vectors
glove = {}
f = open('glove.6B.' + str(WORD_VECTOR_SIZE) + 'd.txt', 'rb')
reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
for row in reader:
    key = row[0]
    vector = map(float, row[1:])
    glove[key] = np.array(vector, dtype=np.float32).reshape(1,-1)
len(glove)


# # Read Dataset
def text2vec(text):
    tokens = word_tokenize(text.lower())
    textVec = np.array([])
    for tok in tokens:
        textVec = np.append(textVec, glove.get(tok, np.zeros((1,WORD_VECTOR_SIZE), dtype=np.float32)))
    return textVec.reshape(1, -1)

def answerpos(context, answer, answer_start):
    start = len(word_tokenize(context[:answer_start]))
    ans_len = len(word_tokenize(answer))
    
    return start, start + ans_len - 1


train = []
for jsonRow in json.loads(open('train.json', 'rb').read()):
    for paragraph in jsonRow['paragraphs']:
        ctxVec = text2vec(paragraph['context'])
        
        for qnaJson in paragraph['qas']:
            qnVec = text2vec(qnaJson['question'])
            
            ansStart, ansEnd = answerpos(paragraph['context'], 
                                           qnaJson['answer']['text'], 
                                           qnaJson['answer']['answer_start'])
            
            train.append((ctxVec, qnVec, ansStart, ansEnd))

def get_batch(i, batch_size, data):
    j = min(i + batch_size, len(data))
    
    ctx = []
    qn = []
    ans_start = []
    ans_end = []
    
    cmax = 0
    qmax = 0
    for k in range(i, j):
        c, q, s, e = data[k]
        ctx.append(c)
        qn.append(q)
        ans_start.append(s)
        ans_end.append(e)
        
        cmax = max(cmax, c.shape[1])
        qmax = max(qmax, q.shape[1])
        
    cVec = np.zeros((len(ctx), cmax), dtype=np.float32)
    qVec = np.zeros((len(ctx), qmax), dtype=np.float32)        
    for i in range(len(ctx)):
        cVec[i, 0:ctx[i].shape[1]] = ctx[i]
        qVec[i, 0:qn[i].shape[1]] = qn[i]
    
    return Variable(cVec),            Variable(qVec),            Variable(np.array(ans_start, dtype=np.int32)).reshape(-1,1),            Variable(np.array(ans_end, dtype=np.int32)).reshape(-1,1)


# # Create Model
opt = optimizers.Adam(alpha=1e-3)
model = SquadNet(WORD_VECTOR_SIZE, H_SIZE, POOL_SIZE, DROPOUT_RATE, USE_GPU)
if USE_GPU:
    model.to_gpu()
opt.setup(model)


# # Define Training Loop
def train_model(model, opt, epoch_start, epoch_end, batch_size, print_interval):
    for epoch in range(epoch_start, epoch_end):
        print "Epoch", epoch + 1, "/", epoch_end
        startTime = time.time()
        epochScore = 0
        epochLoss = 0

        shuffle(train)
        opt.new_epoch()
        
        interval_loss = 0
        interval_size = 0
        interval_start = time.time()
        with chainer.using_config('train', True):
            for i in range(0, len(train), batch_size):
                try:
                    ctx, qn, ans_start, ans_end = get_batch(i, batch_size, train)
                    if USE_GPU:
                        ans_start.to_gpu()
                        ans_end.to_gpu()

                    model.reset_state()
                    pred_start, pred_end = model(ctx, qn)

                    pred_start = pred_start[:ctx.shape[0],:,:]
                    pred_end = pred_end[:ctx.shape[0]]

                    loss_start = F.softmax_cross_entropy(pred_start, ans_start)
                    loss_end = F.softmax_cross_entropy(pred_end, ans_end)
                    loss = loss_start + loss_end

                    interval_loss += loss.data * ctx.shape[0] 
                    interval_size += ctx.shape[0]
                    epochLoss += loss.data * ctx.shape[0] / len(train)
                    if i % print_interval == 0:
                        print i, "/", len(train), ":", interval_loss / interval_size, "(" + str(time.time() - interval_start) + "s)"
                        interval_loss = 0
                        interval_size = 0
                        interval_start = time.time()

                    s = F.argmax(pred_start, axis=1).data
                    e = F.argmax(pred_end, axis=1).data
                    for j in range(s.shape[0]):
                        if s[j] == ans_start.data[j] and e[j] == ans_end.data[j]:
                            epochScore += 1

                    model.cleargrads()
                    loss.backward()

                    opt.update()
                except IndexError as e:
                    print "Error on train index " + str(i) + ":", e
        
        epochAcc = float(epochScore) / len(train)
        
        print "Epoch completed in", time.time() - startTime, "seconds"
        print "Train Acc:", epochAcc, "Train Loss:", epochLoss


# # Train Model
train_model(model, opt, 0, N_EPOCH, MINI_BATCH_SIZE, 1000)
serializers.save_npz('squadnet.model', model)
