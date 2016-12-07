'''
MIT License

Copyright (c) 2017 Sterin, Farrugia, Gripon.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

'''
   This code implements the Vanilla RNN model in theano.
'''

import numpy as np
import random
import theano
import theano.tensor as T
import collections as c
import copy

class RNNv:

   def __init__(self, n_i, n_h, n_o):
      self.n_i = n_i
      self.n_h = n_h
      self.n_o = n_o

      self.rand_init_params()

      self.n_parameters = self.n_h*self.n_i+self.n_h*self.n_h+self.n_h + self.n_o*self.n_h

      self.W_x = theano.shared(copy.deepcopy(self.init_W_x), name='W_x')
      self.W_h = theano.shared(copy.deepcopy(self.init_W_h), name='W_h')
      self.W_y = theano.shared(copy.deepcopy(self.init_W_y), name='W_y')
      self.b1 = theano.shared(np.zeros(self.n_h), name='b')
      self.b2 = theano.shared(np.zeros(self.n_o), name='b')

      self.params = [self.W_x,self.W_h,self.W_y,self.b1, self.b2]

      self.__theano_build__()

   def rand_init_params(self):
      self.init_W_x = np.random.randn(self.n_h,self.n_i)
      self.init_W_h = np.random.randn(self.n_h,self.n_h)
      self.init_W_y = np.random.randn(self.n_o,self.n_h)

   def __theano_build__(self):
      W_x, W_h, W_y, b1, b2 = self.W_x, self.W_h, self.W_y, self.b1, self.b2

      #First dim is time
      x = T.matrix()
      #target
      t = T.matrix()
      #initial hidden state
      h0 = T.vector()



      def step(x_t, h_tm1, W_x, W_h, W_y, b1, b2):
         h_t = T.nnet.sigmoid(T.dot(W_x, x_t)+T.dot(W_h, h_tm1)+b1)
         y_t = T.dot(W_y, h_t)+b2
         return y_t, h_t

      [y, h], _ = theano.scan(step,
                              sequences=x,
                              non_sequences=[W_x,W_h,W_y, b1, b2],
                              outputs_info=[None, h0])


      error = ((y - t) ** 2).sum()
      #error = T.mean(T.nnet.binary_crossentropy(y, t))
      gW_x, gW_h, gW_y, gW_b1,gW_b2 = T.grad(error, [W_x, W_h, W_y, b1, b2])



      self.model = theano.function([x, h0], (y,h))
      self.get_error = theano.function([x, t, h0], error)
      self.bptt = theano.function([x, t, h0], [gW_x, gW_h, gW_y, gW_b1, gW_b2])


      lr = T.scalar()
      self.train_step = theano.function([h0, x, t, lr],
                                        (y, h, error),
                                        updates=c.OrderedDict({W_x: W_x - lr * gW_x,
                                                               W_h: W_h - lr * gW_h,
                                                               W_y: W_y - lr * gW_y,
                                                               b1: b1 - lr*gW_b1,
                                                               b2: b2 - lr*gW_b2}))

   def reset(self, random_init=True):
      if random_init:
         self.rand_init_params()
      self.W_x.set_value(self.init_W_x)
      self.W_h.set_value(self.init_W_h)
      self.W_y.set_value(self.init_W_y)
      self.b1.set_value(np.zeros(self.n_h))
      self.b2.set_value(np.zeros(self.n_o))