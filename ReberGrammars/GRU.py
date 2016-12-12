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
   This code implements the GRU RNN model in theano.
'''

import numpy as np
import random
import theano
import theano.tensor as T
import collections as c
import copy

class GRU:

   def __init__(self, n_i, n_h, n_o):
      self.n_i = n_i
      self.n_h = n_h
      self.n_o = n_o


      self.rand_init_params()

      self.n_parameters = 3*self.n_h*self.n_i+3*self.n_h*self.n_h + self.n_o*self.n_h

      self.W_z = theano.shared(copy.deepcopy(self.init_W_z), name='W_z')
      self.U_z = theano.shared(copy.deepcopy(self.init_U_z), name='U_z')
     
      self.W_r = theano.shared(copy.deepcopy(self.init_W_r), name='W_r')
      self.U_r = theano.shared(copy.deepcopy(self.init_U_r), name='U_r')

      self.W_hp = theano.shared(copy.deepcopy(self.init_W_hp), name='W_h')
      self.U_hp = theano.shared(copy.deepcopy(self.init_W_up), name='U_h')

      self.W_y = theano.shared(copy.deepcopy(self.init_W_y), name='W_y')

      self.b1 = theano.shared(np.zeros(self.n_h), name='b1')
      self.b2 = theano.shared(np.zeros(self.n_h), name='b2')
      self.b3 = theano.shared(np.zeros(self.n_h), name='b3')

      self.params = [self.W_z,self.U_z,self.W_r,self.U_r, self.W_hp, self.U_hp, self.W_y, self.b1, self.b2, self.b3]

      self.__theano_build__()

   def rand_init_params(self):
      self.init_W_z = np.random.randn(self.n_h,self.n_i)
      self.init_U_z = np.random.randn(self.n_h,self.n_h)
      
      self.init_W_r = np.random.randn(self.n_h,self.n_i)
      self.init_U_r = np.random.randn(self.n_h,self.n_h)

      self.init_W_hp = np.random.randn(self.n_h,self.n_i)
      self.init_W_up = np.random.randn(self.n_h,self.n_h)

      self.init_W_y = np.random.randn(self.n_o,self.n_h)

   def __theano_build__(self):
      params = self.params

      #First dim is time
      x = T.matrix()
      #target
      t = T.matrix()
      #initial hidden state
      s0 = T.vector()


      def step(x_t, s_tm1, W_z, U_z, W_r, U_r, W_h, U_h, W_y, b1, b2, b3):
         z = T.nnet.sigmoid(W_z.dot(x_t)+U_z.dot(s_tm1)+b1)
         r = T.nnet.sigmoid(W_r.dot(x_t)+U_r.dot(s_tm1)+b2)
         h = T.tanh(W_h.dot(x_t)+U_h.dot(s_tm1*r)+b3)
         s_t = (1-z)*h + z*s_tm1
         y_t = W_y.dot(s_t)
         return y_t, s_t,z,r

      [y,s,z,r], _ = theano.scan(step,
                              sequences=x,
                              non_sequences=params,
                              outputs_info=[None, s0, None, None])


      error = ((y - t) ** 2).sum()      
      grads = T.grad(error, params)



      self.model = theano.function([x, s0], (y,s,z,r))
      self.get_error = theano.function([x, t, s0], error)
      self.bptt = theano.function([x, t, s0], grads)


      lr = T.scalar()

      chgt = {}
      for i in range(len(params)):
         chgt[params[i]] = params[i]-lr*grads[i]

      self.train_step = theano.function([s0, x, t, lr],
                                        (y, s, error),
                                        updates=c.OrderedDict(chgt))

   def reset(self, random_init=True):
      if random_init:
         self.rand_init_params()

      self.W_z.set_value(self.init_W_z)
      self.U_z.set_value(self.init_U_z)
     
      self.W_r.set_value(self.init_W_r)
      self.U_r.set_value(self.init_U_r)

      self.W_hp.set_value(self.init_W_hp)
      self.U_hp.set_value(self.init_W_up)

      self.W_y.set_value(self.init_W_y)

      self.b1 = theano.shared(np.zeros(self.n_h), name='b1')
      self.b2 = theano.shared(np.zeros(self.n_h), name='b2')
      self.b3 = theano.shared(np.zeros(self.n_h), name='b3')


