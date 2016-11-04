from copy import deepcopy
from itertools import chain

from nltk import Tree as NLTKTree
from nltk import ImmutableTree

"""
def NodePatternNumSubtrees(node_pattern):
  assert node_pattern in node_to_num_subtrees, \
    'Node pattern {0} is not present in dictionary {1}'\
    .format(node_pattern, node_to_num_subtrees)
  return node_to_num_subtrees[node_pattern] 

def NodeToNumSubtrees(tree):
  node_to_num_subtrees = {}
  for subtree in tree.subtrees():
    num_subtrees = len([t for t in subtree.subtrees()])
    node_to_num_subtrees[subtree.node] = num_subtrees
  return node_to_num_subtrees

def RelabelNodes(tree, index = 0):
  if not isinstance(tree, NLTKTree):
    return index
  tree.node = tree.node + '|||' + str(index)
  current_index = index + 1
  for child in tree:
    current_index = RelabelNodes(child, current_index)
  return current_index

node_to_num_subtrees = {}

def AddNumSubtreesFeatStruct(tree):
  global node_to_num_subtrees
  RelabelNodes(tree)
  node_to_num_subtrees = NodeToNumSubtrees(tree)
  tree_feat_struct = Tree.parse(tree.pprint(), node_pattern=NodePatternNumSubtrees)
  return tree_feat_struct
"""

def IsPlausibleEntityPhrase(treep):
  """
  Returns True if the treep is rooted at:
  * NP, WHNP, or
  * NN, NNS, NNP, NNPS that have no direct NP.
  """
  if treep.IsString():
    return False
  if treep.GetRoot() in ['NP', 'WHNP']:
    return True
  if treep.GetRoot() in ['NN', 'NNS', 'NNP', 'NNPS']:
    if len(treep.path) > 1 and \
       get_top(treep.tree[treep.path[:-1]]) not in ['NP', 'WHNP']:
      return True
  return False

def GetPathToLeavesIndices(tree):
  leaf_paths = tree.treepositions('leaves')
  lp2ind = {p : i for i, p in enumerate(leaf_paths)}
  paths = tree.treepositions()
  path_to_leaf_ind = {}
  for path in paths:
    child_leaf_paths = [lp2ind[lp] for lp in leaf_paths if tstartswith(lp, path)]
    path_to_leaf_ind[path] = child_leaf_paths
  return path_to_leaf_ind

def tree_to_xml(tree):
  """
  Returns string XML representation of Tree, in Moses format. E.g.:
  (NP (DT the) (NN house))
  is converted to (without newlines):
  <tree label="NP">
    <tree label="DT">the</tree>
    <tree label="NN">house</tree>
  </tree>
  """
  if IsString(tree):
    return tree
  if any(map(IsString, tree)) and len(tree) > 1:
    raise ValueError('Leaves should be attached to preterminals: {0}'.format(tree))
  return '<tree label="{0}">{1}</tree>'.format(
    tree.label(), ''.join(map(tree_to_xml, tree)))

class Tree(NLTKTree):
  def __init__(self, node, children=None):
    NLTKTree.__init__(self, node, children)
    self.num_subtrees = {}
    self.path_to_leaves = {}
    self.path_to_nodes = {}
    self.path_to_inner_nodes = {}
    self.path_to_inner_nodes_indices = {}
    # Pre-cache leaves lazily. These attributes will be set when needed.
    self.path_to_leaves_indices = None
    self.path_to_num_leaves = None
    self._leaves_cached = None
    self._leaf_to_index = None

  @property
  def leaf_to_index(self):
    if self._leaf_to_index is None:
      self._leaf_to_index = {leaf : i for i, leaf in enumerate(self.leaves_cached)}
    return self._leaf_to_index

  @property
  def leaves_cached(self):
    if self._leaves_cached is None:
      self._leaves_cached = self.leaves()
    return self._leaves_cached

  def __repr__(self):
    return self.pprint(margin=100000).encode('utf-8')

  def __str__(self):
    return repr(self)

  def __getitem__(self, key):
    return super(Tree, self).__getitem__(key)

  def __hash__(self):
    return hash(repr(self))

  def __hash2__(self):
    return hash(id(self))

  def GetPathsSpanningLeaves(self, min_leaf_index, max_leaf_index, path = ()):
    """
    Get all paths that are subpaths of "path" such that they span leaves
    whose indices are between min_leaf_index and max_leaf_index.
    """
    if self.path_to_leaves_indices is None:
      self.path_to_leaves_indices = GetPathToLeavesIndices(self)
    paths = [p for p in self.treepositions() if tstartswith(p, path)]
    paths_result = []
    for p in paths:
      p_leaves_inds = self.path_to_leaves_indices[p]
      if p_leaves_inds[0] >= min_leaf_index \
        and p_leaves_inds[-1] <= max_leaf_index:
        paths_result.append(p)
    return paths_result

  def GetNumSubtrees(self, path = ()):
    if path not in self.num_subtrees:
      if IsString(self[path]):
        self.num_subtrees[path] = 0
      else:
        self.num_subtrees[path] = len([t for t in self[path].subtrees()])
    return self.num_subtrees[path]

  def GetLeaves(self, path = ()):
    if path not in self.path_to_leaves:
      indexed_tree = tree_index(self, path)
      if IsString(indexed_tree):
        leaves = [] if IsVariable(indexed_tree) else [indexed_tree]
      else:
        leaves = indexed_tree.leaves()
      self.path_to_leaves[path] = leaves
    return self.path_to_leaves[path]

  def GetNumLeaves(self, path = ()):
    if self.path_to_leaves_indices is None:
      self.path_to_leaves_indices = GetPathToLeavesIndices(self)
    if self.path_to_num_leaves is None:
      self.path_to_num_leaves = \
        {path : float(len(indices)) \
           for path, indices in self.path_to_leaves_indices.items()}
    assert path in self.path_to_num_leaves
    return self.path_to_num_leaves[path]

  def GetNodes(self, path = ()):
    if path not in self.path_to_nodes:
      indexed_tree = tree_index(self, path)
      if IsString(indexed_tree):
        nodes = [indexed_tree]
      else:
        paths = indexed_tree.treepositions()
        nodes = [get_top(indexed_tree[p]) for p in paths]
      self.path_to_nodes[path] = nodes
    return self.path_to_nodes[path]

  def GetInnerNodes(self, path = ()):
    if path not in self.path_to_inner_nodes:
      indexed_tree = tree_index(self, path)
      if IsString(indexed_tree):
        nodes = []
      else:
        leaves_paths = indexed_tree.treepositions('leaves')
        paths = indexed_tree.treepositions()
        nodes = [get_top(indexed_tree[p]) for p in paths if p not in leaves_paths]
      self.path_to_inner_nodes[path] = nodes
    return self.path_to_inner_nodes[path]

  def GetInnerNodesIndices(self, path = ()):
    """
    Returns a list of tuples with the top of a node and its path
    (which acts as a index).
    """
    if path not in self.path_to_inner_nodes_indices:
      indexed_tree = tree_index(self, path)
      if IsString(indexed_tree):
        nodes_indices = []
      else:
        leaves_paths = indexed_tree.treepositions('leaves')
        paths = indexed_tree.treepositions()
        nodes_indices = [(get_top(indexed_tree[p]), path + p) \
                   for p in paths if p not in leaves_paths]
      self.path_to_inner_nodes_indices[path] = nodes_indices
    return self.path_to_inner_nodes_indices[path]

def GetChildrenPaths(tree, path, max_depth=1):
  """
  Returns the absolute paths of tree below path, up to a maximum
  depth max_depth. If max_depth is set to None, then all subpaths
  are retrieved.
  """
  if IsString(tree):
    return []
  subtree = tree_index(tree, path)
  if IsString(subtree):
    return []
  paths = [path + p for p in subtree.treepositions() if 0 < len(p) <= max_depth]
  return paths

class TreePattern:
  """
  Class to implement a partial subtree. E.g.
  self.tree = (NP (DT the) (NN house))
  self.path = (0,)
  self.subpaths = [(0, 0)]
  would represent the partial subtree (DT ?x0)
  """
  def __init__(self, tree, path, subpaths):
    if isinstance(tree, ImmutableTree):
      self.tree = Tree.fromstring(tree.pprint(margin=10000))
    else:
      self.tree = tree
    self.path = path
    self.subpaths = subpaths
    self.leaves = None
    self.nodes = { True : None, False: None }
    self.inner_nodes = None
    self.excluded_leaves = None
    self.excluded_nodes = None
    self.num_subtrees = None
    self.pattern = None
    self.leaves_indices = None
    self.excluded_leaves_indices = None

  def __hash__(self):
    if self.pattern == None:
      self.ObtainTreePattern()
    return hash(repr(self.pattern))

  def __str__(self):
    return repr(self)

  def __repr__(self):
    if self.pattern == None:
      self.ObtainTreePattern()
    if IsString(self.pattern):
      return self.pattern.encode('utf-8')
    else:
      return self.pattern.pprint(margin=1000).encode('utf-8')

  def __eq__(self, other):
    return repr(other) == repr(self)

  def __ne__(self, other):
      return not self.__eq__(other)

  def HasVariables(self):
    return len(self.subpaths) > 0

  def GetNumVariables(self):
    return len(self.subpaths)

  def GetRoot(self):
    if IsString(self.tree) or \
       IsString(self.tree[self.path]) or \
       [self.path] == self.subpaths:
      return ''
    return get_top(self.tree[self.path])

  # TODO: change name into ObtainTreeRepresentation.
  def ObtainTreePattern(self):
    """
    Obtain tree representation of this tree pattern, where subpaths
    are replaced by numbered variables with a type if they are inner
    nodes, or with empty types if they replace tokens.
    """
    if self.pattern != None:
      return self.pattern
    tree = self.tree
    path = self.path
    subpaths = self.subpaths
    assert all([tstartswith(s, path) for s in subpaths]), \
      'Path {0} is not prefix of subpaths {1} at tree: {2}.'\
      .format(path, subpaths, self.tree)
    subtree = tree_index(tree, path)
    if not subpaths:
      self.pattern = deepcopy(subtree)
    elif IsString(subtree) and IsVariable(subtree):
      self.pattern = subtree
    elif not isinstance(subtree, NLTKTree) and (subpaths[0] == () or path == subpaths[0]):
      self.pattern = '?x0|'
    elif isinstance(subtree, NLTKTree) and (subpaths[0] == () or path == subpaths[0]):
      self.pattern = '?x0|' + get_top(subtree)
    elif not isinstance(subtree, NLTKTree) and subpaths[0] != ():
      raise(ValueError('Tree "{0}" cannot be indexed by {1} from path {2}'\
                       .format(self.tree, subpaths, path)))
    else:
      depth_subtree = len(path)
      tree_pattern = deepcopy(subtree)
      for i, subpath in enumerate(subpaths):
        subpath_relative = subpath[depth_subtree:]
        branch = tree_index(tree, subpath)
        if not isinstance(branch, NLTKTree):
          tree_pattern[subpath_relative] = '?x' + str(i) + '|'
        else:
          tree_pattern[subpath_relative] = '?x' + str(i) + '|' + get_top(branch)
      self.pattern = tree_pattern
    return self.pattern

  def GetNumSubtrees(self):
    if IsString(self.tree) or IsString(self.tree[self.path]):
      return 0
    if not self.num_subtrees:
      self.num_subtrees = \
        self.tree.GetNumSubtrees(self.path) \
        - sum([self.tree.GetNumSubtrees(s) for s in self.subpaths \
               if not IsString(self.tree[s])])
    return self.num_subtrees

  def IsString(self):
    if IsString(self.tree) \
       or (IsString(self.tree[self.path]) and tuple(self.subpaths) != (self.path,)):
      return True
    return False

  def GetExcludedLeaves(self):
    if self.excluded_leaves != None:
      return self.excluded_leaves
    self.excluded_leaves = []
    sorted_subpaths = sorted(self.subpaths)
    for subpath in sorted_subpaths:
      subpath_leaves = GetLeaves(self.tree, subpath)
      self.excluded_leaves.extend(subpath_leaves)
    return self.excluded_leaves

  def GetExcludedLeavesIndices(self):
    if self.excluded_leaves_indices != None:
      return self.excluded_leaves_indices
    if self.tree.path_to_leaves_indices is None:
      self.tree.path_to_leaves_indices = GetPathToLeavesIndices(self.tree)
    if IsString(self.tree):
      self.excluded_leaves_indices = [0]
    else:
      sorted_subpaths = sorted(self.subpaths)
      self.excluded_leaves_indices = list(
        chain(*[self.tree.path_to_leaves_indices[s] for s in sorted_subpaths]))
    return self.excluded_leaves_indices

  def GetExcludedNodes(self):
    if self.excluded_nodes != None:
      return self.excluded_nodes
    self.excluded_nodes = []
    sorted_subpaths = sorted(self.subpaths)
    for subpath in sorted_subpaths:
      indexed_tree = tree_index(self.tree, subpath)
      subpath_nodes = GetNodes(indexed_tree)
      self.excluded_nodes.extend(subpath_nodes)
    return self.excluded_nodes

  def GetLeavesIndices(self):
    if self.leaves_indices != None:
      return self.leaves_indices
    if IsString(self.tree):
      if not self.subpaths:
        self.leaves_indices = [0]
      else:
        assert list(self.subpaths) == [()]
        self.leaves_indices = []
      return self.leaves_indices
    if self.tree.path_to_leaves_indices is None:
      self.tree.path_to_leaves_indices = GetPathToLeavesIndices(self.tree)
    path_leaves_indices = self.tree.path_to_leaves_indices[self.path]
    subpath_leaves_indices = \
      chain(*[self.tree.path_to_leaves_indices[s] for s in self.subpaths])
    self.leaves_indices = set(path_leaves_indices) - set(subpath_leaves_indices)
    self.leaves_indices = sorted(list(self.leaves_indices))
    return self.leaves_indices 

  def GetLeaves(self):
    if self.leaves != None:
      return self.leaves
    elif IsString(self.tree):
      if list(self.subpaths) != [()]:
        self.leaves = [self.tree]
      elif list(self.subpaths) == [()]:
        self.leaves = []
    elif IsString(self.tree[self.path]):
      if list(self.subpaths) != [self.path]:
        self.leaves = [self.tree[self.path]]
      elif list(self.subpaths) == [self.path]:
        self.leaves = []
    else:
      if self.tree.path_to_leaves_indices is None:
        self.tree.path_to_leaves_indices = GetPathToLeavesIndices(self.tree)
      path_leaves_indices = self.tree.path_to_leaves_indices[self.path]
      subpath_leaves_indices = \
        list(chain(*[self.tree.path_to_leaves_indices[p] for p in self.subpaths]))
      self.leaves = [self.tree.leaves_cached[i] for i in path_leaves_indices \
                       if not i in subpath_leaves_indices]
    return self.leaves

  def GetNodes(self, only_leaves=False, only_inner=False):
    """
    Returns the nodes or leaves of the tree pattern (excluding the nodes
    or leaves that are below the variables).
    """
    if (only_leaves, only_inner) in self.nodes:
      return self.nodes[(only_leaves, only_inner)]
    elif IsString(self.tree) and list(self.subpaths) != [()]:
      self.nodes[(only_leaves, only_inner)] = \
        [self.tree] if not only_inner else []
    elif IsString(self.tree) and list(self.subpaths) == [()]:
      self.nodes[(only_leaves, only_inner)] = []
    elif IsString(self.tree[self.path]) and self.subpaths != [self.path]:
      self.nodes[(only_leaves, only_inner)] = \
        [self.tree[self.path]] if not only_inner else []
    elif IsString(self.tree[self.path]) and self.subpaths == [self.path]:
      self.nodes[(only_leaves, only_inner)] = []
    else:
      if only_leaves:
        positions = self.tree.treepositions('leaves')
      else:
        positions = self.tree.treepositions()
        if only_inner:
          leaf_positions = self.tree.treepositions('leaves')
          positions = [p for p in positions if p not in leaf_positions]
      tree_positions = \
        [t for t in positions \
         if tstartswith(t, self.path) \
         and not any(map(lambda s: tstartswith(t, s), self.subpaths))]
      self.nodes[(only_leaves, only_inner)] = \
        [get_top(self.tree[p]) for p in tree_positions]
    return self.nodes[(only_leaves, only_inner)]

  def GetInnerNodes(self):
    """
    Returns the inner nodes of the tree pattern (excluding the inner nodes
    that are below the variables). Leaf nodes are excluded.
    """
    if self.inner_nodes:
      return self.inner_nodes
    if IsString(self.tree):
      self.inner_nodes = []
    else:
      nodes = self.tree.GetInnerNodesIndices(self.path)
      subpath_nodes = list(chain(*[self.tree.GetInnerNodesIndices(s) \
                                     for s in self.subpaths]))
      self.inner_nodes = [n[0] for n in nodes if n not in subpath_nodes]
    return self.inner_nodes

  def GetNumNodes(self):
    if (False, False) not in self.nodes:
      self.GetNodes()
    return len(self.nodes[(False, False)])

  def subtrees(self):
    subtree_positions = \
      [t for t in self.tree.treepositions() \
       if tstartswith(t, self.path) \
       and not (IsString(self.tree[t]) \
                or any(map(lambda s: tstartswith(t, s), self.subpaths)))]
    subtrees = [self.tree[p] for p in subtree_positions]
    return subtrees

  def treepositions(self):
    if IsString(self.tree):
      return [()]
    tree_positions = \
      [t for t in self.tree.treepositions() \
       if tstartswith(t, self.path) \
       and (t in self.subpaths \
            or not any(map(lambda s: tstartswith(t, s), self.subpaths)))]
    return tree_positions

  def __getitem__(self, path):
    return self.tree[path]

def MakeTreePatternOrString(tree, path, subpaths):
  if not IsString(tree):
    return TreePattern(tree, path, subpaths)
  return tree

def tree_index(tr, path):
  if tr == None:
    return None
  if IsString(tr) and len(path) > 0:
    raise(ValueError, \
          "can't index into a string {0} with path {1}".format(tr, str(path)))
  if IsString(tr):
    return tr
  else:
    return tr[path]

def GetPosFromStr(string):
  assert IsString(string)
  return GetVarType(string) if IsVariable(string) else string

def GetPosAt(tr, path):
  if IsString(tr):
    return GetPosFromStr(tr)
  top = get_top(tree_index(tr, path))
  return GetPosFromStr(top)

def get_top(tr):
    """Given a thing that might be a tree or maybe a terminal string, return
    the 'top' of it -- either the node of a tree, or just the string itself."""
    return (tr if IsString(tr) else tr.label())

def IsVarTyped(variable_str):
  assert '|' in variable_str and variable_str.startswith('?x'), \
    'Variable has no correct ortography: {0}'.format(variable_str)
  var_type = ''.join(variable_str.split('|')[1:])
  if var_type != '':
    return True
  return False

def GetVarType(variable_str):
  assert '|' in variable_str and variable_str.startswith('?x'), \
    'Variable has no correct ortography: {0}'.format(variable_str)
  return '|'.join(variable_str.split('|')[1:])

def GetVarName(variable_str):
  assert '|' in variable_str and variable_str.startswith('?x'), \
    'Variable has no correct ortography: {0}'.format(variable_str)
  return variable_str.split('|')[0]

def UntypeVar(variable_str):
  assert '|' in variable_str and variable_str.startswith('?x'), \
    'Variable has no correct ortography: {0}'.format(variable_str)
  return variable_str.split('|')[0] + '|'

def UntypeVars(tree):
  """
  Given a tree, possibly with variables in its leaves, return the same tree
  where all variables are untyped.
  """
  if IsString(tree) and IsVariable(tree) and IsTypedVar(tree):
    return UntypeVar(tree)
  vars_paths = variables_to_paths(tree)
  for var, path in vars_paths:
    tree[path] = UntypeVar(var)
  return tree

def IsVariable(variable_str):
  return '|' in variable_str \
         and variable_str.startswith('?x')

def variables_to_paths(tr):
  """Given a tree, return a list of (var,path) tuples -- not a dictionary
  because a variable may occur more than once."""
  out = []
  if IsString(tr):
    if tr.startswith('?x'):
      out.append((tr,()))
  else:
    for path in tr.treepositions('leaves'):
      leaf = tr[path]
      if leaf.startswith('?x'):
        out.append((leaf, path))
  return out

def variables_to_paths_(tr):
  """Given a tree, return a list of (var,path) tuples -- not a dictionary
  because a variable may occur more than once."""
  out = []
  if IsString(tr):
    if tr.startswith('?x'):
      out.append((tr,()))
  else:
    paths = [path for path in tr.treepositions() \
                    if get_top(tr[path]).startswith('?x')]
    out = [(get_top(tr[path]), path) for path in paths]
  return out

def Quote(token):
  unquoted_token = token.strip('"')
  return '"' + unquoted_token + '"'

def Unquote(token):
  unquoted_token = token.strip('"')
  return unquoted_token

def GetTreeString(tree):
  if IsString(tree):
    return tree
  return tree.pprint(margin=10000)

def GetLeaves(tree, path = ()):
  if IsString(tree):
    return [tree]
  elif isinstance(tree, TreePattern):
    return tree.GetLeaves()
  return tree.GetLeaves(path)

def GetNodes(tree, path = ()):
  if IsString(tree):
    return [tree]
  elif isinstance(tree, TreePattern):
    return tree.GetNodes()
  return tree.GetNodes(path)

def immutable(tr):
  """Given a Tree, make it an ImmutableTree."""
  if IsString(tr):
    return tr
  elif isinstance(tr, TreePattern):
    return tr.tree.freeze()
  return tr.freeze()

def tstartswith(tuple_a, tuple_b):
  """
  Returns true if tuple_a starts with tuple_b.
  In other words, it returns true if tuple_b is a prefix
  of tuple_a.
  """
  return tuple_a[:len(tuple_b)] == tuple_b

def IsString(variable):
  return isinstance(variable, str) or isinstance(variable, unicode)

def BinarizeTreeString(tree_str):
  """
  Convert a string representation of a tree into its Chomsky Normal Form.
  Note: It returns a string representation of the binarized tree.
  """
  tree = Tree.fromstring(tree_str)
  tree.chomsky_normal_form()
  tree_str = tree.pprint(margin=10000)
  return tree_str

def LabelAndRank(tree):
  label, rank = None, None
  if IsString(tree):
    if not IsVariable(tree):
      label, rank = get_top(tree), 0
    else:
      rank = None
      label = GetVarType(tree) if IsVarTyped(tree) else None
  else:
    label, rank = get_top(tree), len(tree)
  return label, rank

def LabelAndTypedRank(tree):
  if not isinstance(tree, NLTKTree):
    tree_pos = tree
    tree_branches_pos = tuple([tree])
  else:
    tree_pos = get_top(tree)
    tree_branches_pos = []
    for t in tree:
      if isinstance(t, NLTKTree):
        tree_branches_pos.append(get_top(t))
      else:
        tree_branches_pos.append('')
    tree_branches_pos = tuple(tree_branches_pos)
  return (tree_pos, tree_branches_pos)

def TreeContains(tree, subtree):
  # Subtree is a variable, and matches everything.
  subtree_top = get_top(subtree)
  tree_top = get_top(tree)
  if subtree_top.startswith('?x'):
    # Get type of the variable.
    var_type = '|'.join(subtree_top.split('|')[1:])
    if var_type == '' or tree_top == var_type:
      return True
    else:
      return False

  tree_is_inst_nltk = isinstance(tree, NLTKTree)
  subtree_is_inst_nltk = isinstance(subtree, NLTKTree)
  # Both are strings and one of them is a QA variable "[]"
  if (not tree_is_inst_nltk) and \
     (tree_top == "[]" or subtree_top == "[]"):
    return True

  # tree and subtree are different types, or they have different POS tag,
  # or they have different number of children.
  if tree_is_inst_nltk and not subtree_is_inst_nltk \
     or (not tree_is_inst_nltk and subtree_is_inst_nltk) \
     or tree_top != subtree_top \
     or len(tree) != len(subtree):
    return False

  # Both are strings and equal to each other.
  if (not tree_is_inst_nltk) and tree_top == subtree_top:
    return True

  # Both are trees, and their subtrees are equal.
  for i, src_branch in enumerate(tree):
    trg_branch = subtree[i]
    if not TreeContains(src_branch, trg_branch):
      return False
  return True

def tree_or_string(s):
  """Given a string loaded from the yaml, produce either a Tree or a string,
  if it's just a terminal."""
  if s.startswith(u"("):
    return Tree.fromstring(s)
  return s

