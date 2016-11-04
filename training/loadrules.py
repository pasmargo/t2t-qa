import codecs
from collections import defaultdict
import simplejson
import yaml

from training.transductionrule import XTRule, ParseTiburonRule
from utils.tree_tools import Tree, tree_or_string

def LoadRulesJson(infile, num_occur = 0):
  # Read all json records.
  data_freq = defaultdict(int)
  for line in infile.readlines():
    if not line.strip('\n'):
      continue
    data_freq[line.strip('\n')] += 1
  data = [simplejson.loads(key) \
            for (key, value) in data_freq.items() if value > num_occur]
  # Filter in all records that are rules.
  loaded = [d['rule'] for d in data if 'rule' in d]
  return loaded

def LoadRulesYaml(infile, num_occur = 0):
  loaded = yaml.load(infile, Loader=yaml.CLoader)
  return loaded

def LoadRulesTiburon(infile, num_occur = 0):
  data_freq = defaultdict(int)
  for line in infile.readlines():
    data_freq[line.strip('\n')] += 1
  data = [ParseTiburonRule(key) \
            for (key, value) in data_freq.items() if value > num_occur]
  # Filter in all records that are rules.
  loaded = [d for d in data if d is not None]
  return loaded

def loadrules(fn, fmt = 'json', num_occur = 0):
  """
  Given a filename fn of a file containing transducer rules (XTRule),
  it parses the file in 'json' (default), 'yaml' or 'tiburon', and returns
  a list of XTRules.
  """
  out = []
  loaded = None
  with codecs.open(fn, 'r', 'utf-8') as infile:
    if fmt == 'json':
      loaded = LoadRulesJson(infile, num_occur)
    elif fmt == 'yaml':
      loaded = LoadRulesYaml(infile, num_occur)
    elif fmt == 'tiburon':
      loaded = LoadRulesTiburon(infile, num_occur)
  # if not loaded:
  #   raise ValueError("No rules loaded from file: {0} with format {1}".format(fn, fmt))

  for d in loaded:
    lhs = tree_or_string(d["lhs"].strip('"'))
    rhs = tree_or_string(d["rhs"].strip('"'))
    state = d["state"]

    weight = d.get("weight", 1.0)
    weight = float(weight) 

    if "newstates" in d:
      if fmt == 'tiburon':
        newstates = d["newstates"]
      else:
        newstates = paths_as_dicts(d["newstates"])
    else:
      newstates = {}

    # Parameter tying.
    tied_to = d.get("tied_to", None)
    # Features.
    features = d.get("features", None)

    newrule = XTRule(state, lhs, rhs, newstates, weight)
    newrule.tied_to = tied_to
    newrule.features = features

    out.append(newrule)
  return list(set(out))

def GetInitialStateJson(infile):
  for line in infile.readlines():
    data = simplejson.loads(line.strip('\n'))
    if 'general_info' in data and 'initial_state' in data['general_info']:
      return data['general_info']['initial_state']
  return None

def GetInitialStateYaml(infile):
  for line in infile.readlines():
    data = line.strip('\n').split()
    if len(data) > 2 and data[0] == '#' and data[1] == 'initial_state:':
      initial_state = data[2]
      return initial_state
  return None

def GetInitialStateTiburon(infile):
  for line in infile.readlines():
    data = line.strip('\n')
    if not data.startswith('%') and ' -> ' not in data:
      initial_state = data
      return initial_state
  return None

def GetInitialState(fn, fmt = 'json'):
  """
  Given a filename fn of a file containing transducer rules (XTRule),
  it parses the file in 'json' (default) or 'yaml', and returns a list
  of XTRules.
  """
  initial_state = None
  with codecs.open(fn, 'r', 'utf-8') as infile:
    if fmt == 'json':
      initial_state = GetInitialStateJson(infile)
    elif fmt == 'yaml':
      initial_state = GetInitialStateYaml(infile)
    elif fmt == 'tiburon':
      initial_state = GetInitialStateTiburon(infile)

  if initial_state is None:
    raise ValueError("Initial state not found in {0} file: {1}".format(fmt, fn))

  return initial_state

def paths_as_dicts(fromyaml):
    """Given a list of lists of the shape [[path,state],...], produce a
    dictionary of the shape {path:state, ...}."""
    out = {}
    for (path, state) in fromyaml:
        out[tuple(path)] = state
    return out

