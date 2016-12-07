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
      This code implements  the  Regular and  Embedded 
   Reber's Grammar. It can produces strings  from these 
   grammars and transforms them into vectors' sequences.
'''


import numpy as np

# Automaton graph
graph = {0:{1:'T',4:'P'}, 1:{1:'S',2:'X'}, 2:{3:'S',4:'X'}, 3:{}, 4:{4:'T',5:'V'}, 5:{2:'P',3:'V'}}

translate = {'B':0, 'T':1, 'S':2, 'X':3, 'V':4, 'P':5, 'E':6}

def basis(a):
   b = np.zeros(len(translate))
   b[translate[a]] = 1.0
   return b

# Convert a reber sequence to a vector one
def reber_to_seq(ben):
   seq = []
   for a in ben:
      seq.append(basis(a))
   return seq

# Generates reber's strings over uniform distrib
def get_reber():
   curr_state = 0
   curr_reber = 'B'
   while len(graph[curr_state]) != 0:
      next_state = np.random.choice(graph[curr_state].keys())
      curr_reber += graph[curr_state][next_state]
      curr_state = next_state
   curr_reber += 'E'
   return curr_reber

# Generates embedded reber's strings over uniform distrib
def get_emb_reber():
   core = get_reber()
   cases = [('BT','TE'), ('BP', 'PE')]
   choix = np.random.choice(range(0,2))
   return cases[choix][0]+core+cases[choix][1]