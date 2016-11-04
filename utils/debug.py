from __future__ import print_function
from sys import stderr, stdout

def warning(*objs):
  print("WARNING: ", *objs, file=stderr)

def info(*objs):
  print("INFO: ", *objs, file=stdout)

