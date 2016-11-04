#!/usr/bin/python

import codecs

# Read the tree_string ||| {tree_}string pairs from a file.
# The file contains lines like:
# NP(DT(a) NN(house))
# There is a house (or its tree version)
# NP(DT(An) JJ(Italian))
# At least one Italian (or its tree version)
def LoadCorpus(training_filename):
  finput = codecs.open(training_filename, 'r', 'utf-8')
  pairs = []
  i = 0
  # from pudb import set_trace; set_trace()
  for line in finput.readlines():
    if i % 2 == 0:
      pair = [line.strip(' \t\n\r')]
    else:
      pair.append(line.strip(' \t\n\r'))
      pairs.append(pair)
    i = i + 1
  finput.close()
  return pairs

# Output the transductions into a file.
# The initial state is written in the first line and is "q".
def SaveTransductionsIntoFile(transductions, filename):
  foutput = codecs.open(filename, 'w', 'utf-8')
  # Initial state.
  foutput.write('q\n')
  # Transductions.
  transductions_list = list(transductions)
  transductions_list.sort()
  for t in transductions_list:
    foutput.write('%s\n' % t)
  foutput.close()

