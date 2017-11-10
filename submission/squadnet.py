# # Imports and Settings
import numpy as np

import chainer
from chainer import Chain, Variable, Parameter
from chainer import iterators, optimizers, serializers
import chainer.initializers as I
import chainer.functions as F
import chainer.links as L


# # Define Network
class CoattentionEncoder(Chain):
    def __init__(self, wordvec_size, h_size, dropout_ratio, use_gpu=False):
        super(CoattentionEncoder, self).__init__()
        
        self.wordvec_size = wordvec_size
        self.h_size = h_size
        self.dropout_ratio = dropout_ratio
        self.use_gpu = use_gpu
        
        with self.init_scope():
            self.ctxRU = L.LSTM(wordvec_size, h_size)

            self.qnRU = L.LSTM(wordvec_size, h_size)
            self.qnLinear = L.Linear(h_size, h_size)
            
            self.outFwd = L.LSTM(3*h_size, h_size)
            self.outBwd = L.LSTM(3*h_size, h_size)
            self.outLinear = L.Linear(2*h_size, h_size)
            
            if use_gpu:
                print "CodynamicAttention uses GPU", self.use_gpu
                self.ctxRU.to_gpu()
                self.qnRU.to_gpu()
                self.qnLinear.to_gpu()
                self.outFwd.to_gpu()
                self.outBwd.to_gpu()
                self.outLinear.to_gpu()
            
    def reset_state(self):
        self.ctxRU.reset_state()
        self.qnRU.reset_state()
        self.outFwd.reset_state()
        self.outBwd.reset_state()
        
    def get_para_rep(self, para, ru):
        P = []
        for i in range(0, para.shape[1], self.wordvec_size):
            word = para[:, i:i+self.wordvec_size]
            if self.use_gpu: 
                word.to_gpu()
            P.append(F.dropout(ru(word), self.dropout_ratio))
            
        return F.transpose(F.dstack(P), (0, 1, 2))
            
    def __call__(self, ctx, qn):
        # context representation
        Ds = self.get_para_rep(ctx, self.ctxRU)
        
        #question representation
        Qs = self.get_para_rep(qn, self.qnRU)
        
        out_ins = []
        for i in range(Ds.shape[0]):
            D = Ds[i]
            Q = Qs[i]
            
            #attention
            affinity = F.matmul(D.T, Q)
            A_Q = F.softmax(affinity)
            A_D = F.softmax(affinity.T)

            C_Q = F.matmul(D, A_Q)
            C_D = F.matmul(F.concat((Q, C_Q), axis=0), A_D)
            
            out_ins.append(F.concat((D, C_D), axis=0).T)
        out_ins = F.transpose(F.dstack(out_ins), (0,2,1))

        #output
        h_fwd = []
        for fout in out_ins:
            h_fwd.append(F.dropout(self.outFwd(fout), self.dropout_ratio))
        h_fwd = F.dstack(h_fwd)

        h_bwd = []
        for bout in out_ins[::-1]:
            h_bwd.append(F.dropout(self.outBwd(bout), self.dropout_ratio))
        h_bwd = F.dstack(h_bwd)
        
        u_in = F.transpose(F.concat((h_fwd, h_bwd)), (0,2,1))
        U = F.dropout(self.outLinear(u_in.reshape(-1, 2*self.h_size)), self.dropout_ratio)
        return U.reshape(Ds.shape[0], -1, self.h_size)

class Highway(Chain):
    def __init__(self, h_size, pool_size, dropout_ratio, use_gpu=False):
        super(Highway, self).__init__()
        
        self.h_size = h_size
        self.pool_size = pool_size
        self.dropout_ratio = dropout_ratio
        self.use_gpu = use_gpu
                
        with self.init_scope():
            self.MLP = L.Linear(3*h_size, h_size, nobias=True)
            self.M1 = L.Maxout(2*h_size, h_size, pool_size)
            self.M2 = L.Maxout(h_size, h_size, pool_size)
            self.M3 = L.Maxout(2*h_size, 1, pool_size)
            
            if use_gpu:
                print "Highway uses GPU", self.use_gpu
                self.MLP.to_gpu()
                self.M1.to_gpu()
                self.M2.to_gpu()
                self.M3.to_gpu()
            
    def __call__(self, U, h, us, ue):
        if self.use_gpu:
            U.to_gpu()
            h.to_gpu()
            us.to_gpu()
            ue.to_gpu()
        
        r = F.tanh(self.MLP(F.hstack([h, us, ue])))
        rs = []
        for i in range(U.shape[0]):
            rs.append(F.broadcast_to(r[i], U[i].shape))
        r = F.transpose(F.dstack(rs), (2,0,1))
        
        m_in = F.concat((U, r), axis=2).reshape(-1, 2*self.h_size)
        m1 = F.dropout(self.M1(m_in), self.dropout_ratio)
        m2 = F.dropout(self.M2(m1), self.dropout_ratio)
        m3 = self.M3(F.concat((m1,m2)))
        
        return m3.reshape(U.shape[0], -1, 1)

class DynamicPointingDecoder(Chain):
    def __init__(self, h_size, pool_size, dropout_ratio, use_gpu=False):
        super(DynamicPointingDecoder, self).__init__()
        
        self.dropout_ratio = dropout_ratio
        self.use_gpu = use_gpu
                
        with self.init_scope():
            self.dec_state = L.LSTM(2*h_size, h_size)
            self.HwayStart = Highway(h_size, pool_size, dropout_ratio, use_gpu)
            self.HwayEnd = Highway(h_size, pool_size, dropout_ratio, use_gpu)
            
            if self.use_gpu:
                print "DynamicPointincDecoded uses GPU", self.use_gpu
                self.dec_state.to_gpu()
                self.HwayStart.to_gpu()
                self.HwayEnd.to_gpu()
            
    def reset_state(self):
        self.dec_state.reset_state()
            
    def __call__(self, U, us, ue):
        if self.use_gpu:
            U.to_gpu()
            us.to_gpu()
            ue.to_gpu()
        
        h = F.dropout(self.dec_state(F.concat((us,ue))), self.dropout_ratio)
        alpha = self.HwayStart(U, h, us, ue)
        s = F.argmax(alpha, axis=1).data.reshape(-1)
        beta = self.HwayEnd(U, h, U[range(U.shape[0]), s], ue)
        
        return alpha, beta

class SquadNet(Chain):
    def __init__(self, wordvec_size, h_size, pool_size, dropout_rate, use_gpu=False):
        super(SquadNet, self).__init__()
        self.use_gpu = use_gpu
                
        with self.init_scope():
            self.encoder = CoattentionEncoder(wordvec_size, h_size, dropout_rate, use_gpu)
            self.decoder = DynamicPointingDecoder(h_size, pool_size, dropout_rate, use_gpu)
            
            if use_gpu:
                print "SquadNet uses GPU", self.use_gpu
                self.encoder.to_gpu()
                self.decoder.to_gpu()
            
    def reset_state(self):
        self.encoder.reset_state()
        self.decoder.reset_state()
            
    def __call__(self, ctx, qn): 
        U = self.encoder(ctx, qn)
        
        start = np.zeros(U.shape[0], 'i')
        end = np.zeros(U.shape[0], 'i') - 1        
        for i in range(4):            
            us = U[range(U.shape[0]), start]
            ue = U[range(U.shape[0]), end]
            alpha, beta = self.decoder(U, us, ue)
            
            start = F.argmax(alpha, axis=1).data.reshape(-1)
            end = F.argmax(beta, axis=1).data.reshape(-1)
        return alpha, beta
