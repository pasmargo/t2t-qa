import codecs
import cPickle as pickle
import  simplejson

from itertools import tee
from types import GeneratorType

Tee = tee([], 1)[0].__class__

# Technique to memoize a generator copied from:
# http://stackoverflow.com/questions/4566769/can-i-memoize-a-python-generator
def memoized_gen(f):
  cache={}
  def ret(*args):
    print('Caching generator')
    if args not in cache:
      cache[args]=f(*args)
    if isinstance(cache[args], (GeneratorType, Tee)):
      # the original can't be used any more,
      # so we need to change the cache as well
      cache[args], r = tee(cache[args])
      return r
    return cache[args]
  return ret

def TryPickleLoad(filename):
  """
  Load pickles a dictionary from filename.
  If it fails, it returns an empty dictionary.
  """
  try:
    with open(filename, 'rb') as f:
      data = pickle.load(f)
  except IOError:
    data = {}
  except EOFError:
    data = {}
  return data

def TryPickleDump(data, filename):
  with open(filename, 'wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def TryJsonLoad(filename):
  try:
    f = codecs.open(filename, 'r', 'utf-8')
    data = simplejson.load(f)
    f.close()
  except (IOError, EOFError, simplejson.JSONDecodeError):
    data = {}
  return data

def TryJsonDump(data, filename):
  with codecs.open(filename, 'w', 'utf-8') as f:
    simplejson.dump(data, f, indent=2)

def update_cache_and_save(cache, queue, fname, dump_func):
  """
  Receive from queue items from other processes.
  if there are no items, we assume that the cache has
  not changed and the cache on disk does not need to be updated.
  Otherwise, it saves the updated version on disk fname.
  Items in the queue are tuples (key_tuple, value),
  where key_tuple is a tuple of primitives (e.g strings, integers).
  """
  cache_has_changed = not queue.empty()
  while not queue.empty():
    key_tuple, value = queue.get()
    if key_tuple not in cache:
      cache[key_tuple] = value
  if cache_has_changed:
    dump_func(cache, fname)

