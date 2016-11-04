#!python
#cython: boundscheck=False, wraparound=False

cimport cython
from libc.stdlib cimport malloc, free

def AreDisjointPaths(paths):
  cdef int num_paths, min_length, i, j, k
  cdef int diff_prefix = 0, are_disjoint_paths = 1
  cdef int *path_lengths
  cdef int **paths_array

  # Create array that stores the length of each path.
  num_paths = len(paths)
  path_lengths = <int *>malloc(num_paths * cython.sizeof(int))
  for i in range(num_paths):
    path_lengths[i] = len(paths[i])

  # Create array of arrays that stores list of paths.
  paths_array = <int **>malloc(num_paths * cython.sizeof(cython.p_int))
  for i in range(num_paths):
    paths_array[i] = <int *>malloc(path_lengths[i] * cython.sizeof(int))
    for j in range(path_lengths[i]):
      paths_array[i][j] = paths[i][j]

  # For every path and the rest of the paths, check whether
  # they share a prefix.
  for i in range(num_paths):
    for j in range(i + 1, num_paths):
      if path_lengths[i] >= path_lengths[j]:
        min_length = path_lengths[j]
      else:
        min_length = path_lengths[i]
      diff_prefix = 0
      for k in range(min_length):
        if paths_array[i][k] != paths_array[j][k]:
          diff_prefix = 1
          break
      if diff_prefix == 0:
        are_disjoint_paths = 0
        break
    if are_disjoint_paths == 0:
      break

  # Release the memory that has been reserved.
  for i in range(num_paths):
    free(paths_array[i])
  free(paths_array)
  free(path_lengths)
  if are_disjoint_paths == 0:
    return <bint>0
  else:
    return <bint>1

def GetCommonParentsAt(paths, int max_depth):
  cdef int path_length, num_paths, parent_length, is_common_prefix = 1
  cdef int shortest_path_length = 1000, longest_path_length = -1
  cdef int prefix_length = -1, i, j, k
  cdef int *shortest_path_array
  cdef int **paths_array

  common_parents = [()]
  if not paths:
    return common_parents
  # Convert the list of tuples into a double pointer array.
  num_paths = len(paths)
  paths_array = <int **>malloc(num_paths * cython.sizeof(cython.p_int))
  for i in range(num_paths):
    path_length = len(paths[i])
    paths_array[i] = <int *>malloc(path_length * cython.sizeof(int))
    if path_length < shortest_path_length:
      shortest_path_length = path_length
      shortest_path_array = paths_array[i]
      shortest_path = paths[i]
    if path_length > longest_path_length:
      longest_path_length = path_length
    for j in range(path_length):
      paths_array[i][j] = paths[i][j]

  # If there is only one path, possible prefixes start from the
  # length of such path. If there are more, a common prefix of a
  # *disjoint* set of paths must be strictly shorter than the
  # shortest path.
  if shortest_path_length == 0:
    return common_parents
  if num_paths == 1:
    parent_length = shortest_path_length
  else:
    parent_length = shortest_path_length - 1

  # Find the length of the largest common prefix.
  for prefix_length in range(0, parent_length):
    for j in range(1, num_paths):
      first_path_position = paths_array[0][prefix_length]
      # Check whether a prefix up to position i is common to all paths.
      if paths_array[j][prefix_length] != first_path_position:
        is_common_prefix = 0
        break
    if is_common_prefix == 0:
      prefix_length -= 1
      break
  # At this point, the variable prefix_length contains the length
  # of the largest common prefix to all paths. If the prefix_length == -1,
  # it means that paths had no common prefix. Only the path to the root ()
  # would be common to all of them.
  common_parents = []
  if prefix_length == -1:
    common_parents.append( () )
  else:
    common_parents = [shortest_path[0:k] for k in \
      range(prefix_length + 1, max(-1, longest_path_length - max_depth - 1), -1)]

  # Release the memory that has been reserved.
  for i in range(num_paths):
    free(paths_array[i])
  free(paths_array)
  return common_parents
