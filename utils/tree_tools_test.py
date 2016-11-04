#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from nltk import ImmutableTree

from utils.tree_tools import (Tree, TreePattern, GetChildrenPaths,
  tree_or_string, TreeContains, GetPosAt, tree_to_xml, IsPlausibleEntityPhrase)
# from utils.tree_tools import Tree, TreePattern, AddNumSubtreesFeatStruct

"""
class NumSubtreesTestCase(unittest.TestCase):
  def setUp(self):
    self.tree = Tree.fromstring('(S (NP (DT the) (NN house)) (VP (VBZ is) (JJ beautiful)))')

  def test_SetFeatStruct(self):
    pass
    tree_feat_struct = AddNumSubtreesFeatStruct(self.tree)
    self.assertEqual(7, tree_feat_struct.label()['num_subtrees'])
"""

class IsPlausibleEntityPhraseTestCase(unittest.TestCase):
  def setUp(self):
    self.tree = tree_or_string('(SBARQ (WHNP (WHNP (WDT what) (NN type)) (PP (IN of) (NP (NN tea)))) (SQ (VBZ is) (NP (NN gunpowder) (NN tea))))')

  def test_NegativeString(self):
    path = (1,1,1,0)
    subpaths = []
    tree_pattern = TreePattern(self.tree, path, subpaths)
    self.assertFalse(IsPlausibleEntityPhrase(tree_pattern))

  def test_NegativeNN1(self):
    path = (1,1,1)
    subpaths = []
    tree_pattern = TreePattern(self.tree, path, subpaths)
    self.assertFalse(IsPlausibleEntityPhrase(tree_pattern))

  def test_NegativeNN2(self):
    path = (1,1,0)
    subpaths = []
    tree_pattern = TreePattern(self.tree, path, subpaths)
    self.assertFalse(IsPlausibleEntityPhrase(tree_pattern))

  def test_PositiveNP(self):
    path = (1,1)
    subpaths = []
    tree_pattern = TreePattern(self.tree, path, subpaths)
    self.assertTrue(IsPlausibleEntityPhrase(tree_pattern))

  def test_PositiveWHNP1(self):
    path = (0,)
    subpaths = []
    tree_pattern = TreePattern(self.tree, path, subpaths)
    self.assertTrue(IsPlausibleEntityPhrase(tree_pattern))

  def test_PositiveWHNP2(self):
    path = (0,0)
    subpaths = []
    tree_pattern = TreePattern(self.tree, path, subpaths)
    self.assertTrue(IsPlausibleEntityPhrase(tree_pattern))

  def test_PositiveNP2(self):
    path = (0,1,1)
    subpaths = []
    tree_pattern = TreePattern(self.tree, path, subpaths)
    self.assertTrue(IsPlausibleEntityPhrase(tree_pattern))

  def test_NegativeNN3(self):
    path = (0,1,1,0)
    subpaths = []
    tree_pattern = TreePattern(self.tree, path, subpaths)
    self.assertFalse(IsPlausibleEntityPhrase(tree_pattern))

  def test_NegativePP(self):
    path = (0,1,)
    subpaths = []
    tree_pattern = TreePattern(self.tree, path, subpaths)
    self.assertFalse(IsPlausibleEntityPhrase(tree_pattern))

class ToXmlTestCase(unittest.TestCase):
  def test_Leaf(self):
    tree = tree_or_string('the')
    xml_str = tree_to_xml(tree)
    expected_xml_str = 'the'
    self.assertEqual(expected_xml_str, xml_str)

  def test_Preterminal(self):
    tree = tree_or_string('(DT the)')
    xml_str = tree_to_xml(tree)
    expected_xml_str = '<tree label="DT">the</tree>'
    self.assertEqual(expected_xml_str, xml_str)

  def test_NoPreterminalRaisesValueError(self):
    tree = tree_or_string('(NP the house)')
    with self.assertRaises(ValueError):
      xml_str = tree_to_xml(tree)

  def test_Nonterminal(self):
    tree = tree_or_string('(NP (DT the) (NN house))')
    xml_str = tree_to_xml(tree)
    expected_xml_str = '<tree label="NP"><tree label="DT">the</tree><tree label="NN">house</tree></tree>'
    self.assertEqual(expected_xml_str, xml_str)

class TreeContainsTestCase(unittest.TestCase):
  def test_PositiveMatchTerminal(self):
    tree = tree_or_string('the')
    subtree = tree_or_string('the')
    self.assertTrue(TreeContains(tree, subtree))

  def test_NegativeMatchTerminal(self):
    tree = tree_or_string('the')
    subtree = tree_or_string('da')
    self.assertFalse(TreeContains(tree, subtree))

  def test_PositiveMatchPreterminal(self):
    tree = tree_or_string('(DT the)')
    subtree = tree_or_string('(DT ?x0|)')
    self.assertTrue(TreeContains(tree, subtree))

  def test_NegativeMatchPreterminal(self):
    tree = tree_or_string('(DT the)')
    subtree = tree_or_string('(JJ ?x0|)')
    self.assertFalse(TreeContains(tree, subtree))

  def test_NegativeMatchPreterminalToNonterminal(self):
    tree = tree_or_string('(DT the)')
    subtree = tree_or_string('(NP (JJ ?x0|))')
    self.assertFalse(TreeContains(tree, subtree))

  def test_PositiveMatchNonterminal(self):
    tree = Tree.fromstring('(NP (DT the) (NN house))')
    subtree = tree_or_string('(NP ?x0|DT ?x1|NN)')
    self.assertTrue(TreeContains(tree, subtree))

  def test_NegativeMatchNonterminal(self):
    tree = Tree.fromstring('(NP (DT the) (NN house))')
    subtree = tree_or_string('(VP ?x0|DT ?x1|NN)')
    self.assertFalse(TreeContains(tree, subtree))

  def test_PositiveMatchNonterminalVariable(self):
    tree = Tree.fromstring('(NP (DT the) (NN house))')
    subtree = tree_or_string('?x0|NP')
    self.assertTrue(TreeContains(tree, subtree))

  def test_PositiveMatchNonterminal2Levels(self):
    tree = Tree.fromstring('(NP (DT the) (NN house))')
    subtree = tree_or_string('(NP (DT the) ?x1|NN)')
    self.assertTrue(TreeContains(tree, subtree))

  def test_NegativeMatchNonterminal2Levels(self):
    tree = Tree.fromstring('(NP (DT the) (NN house))')
    subtree = tree_or_string('(NP (DT da) ?x1|NN)')
    self.assertFalse(TreeContains(tree, subtree))

  def test_PositiveMatchNonterminal2LevelsFull(self):
    tree = Tree.fromstring('(NP (DT the) (NN house))')
    subtree = tree_or_string('(NP (DT the) (NN house))')
    self.assertTrue(TreeContains(tree, subtree))

  def test_NegativeMatchNonterminal2LevelsFullLeft(self):
    tree = Tree.fromstring('(NP (DT the) (NN house))')
    subtree = tree_or_string('(NP (DT da) (NN house))')
    self.assertFalse(TreeContains(tree, subtree))

  def test_NegativeMatchNonterminal2LevelsFullRight(self):
    tree = Tree.fromstring('(NP (DT the) (NN house))')
    subtree = tree_or_string('(NP (DT the) (NN condominium))')
    self.assertFalse(TreeContains(tree, subtree))

  def test_PositiveMatchQAVar(self):
    tree = Tree.fromstring('(NP (DT the) (NN house))')
    subtree = tree_or_string('(NP (DT the) (NN []))')
    self.assertTrue(TreeContains(tree, subtree))

  def test_TerminalJapaneseUntypedVar(self):
    tree = tree_or_string(u'学生')
    subtree = tree_or_string(u'?x0|')
    self.assertTrue(TreeContains(tree, subtree))

  def test_PreterminalJapaneseUntypedVar(self):
    tree = tree_or_string(u'(学生 hello)')
    subtree = tree_or_string(u'?x0|')
    self.assertTrue(TreeContains(tree, subtree))

  def test_PreterminalJapaneseTypedVar(self):
    tree = tree_or_string(u'(学生 hello)')
    subtree = tree_or_string(u'?x0|学生')
    self.assertTrue(TreeContains(tree, subtree))

  def test_PreterminalJapaneseTypedVarImmutableTree(self):
    tree = ImmutableTree.fromstring(u'(学生 hello)')
    subtree = tree_or_string(u'?x0|学生')
    self.assertTrue(TreeContains(tree, subtree))

  def test_LeafVarNoMatch(self):
    tree = tree_or_string(u'(NP (DT the) (NN house))')
    subtree = tree_or_string(u'(NP ?x0|DT ?x1|JJ)')
    self.assertFalse(TreeContains(tree, subtree))

class GetChildrenPathsTestCase(unittest.TestCase):
  def setUp(self):
    self.tree = tree_or_string(
      '(S (NP (DT the) (NN house)) (VP (VBZ is) (JJ beautiful)))')

  def test_Terminal(self):
    from_node = (0, 0, 0)
    children_paths = GetChildrenPaths(self.tree, from_node)
    expected_children_paths = []
    self.assertItemsEqual(expected_children_paths, children_paths)

  def test_PreterminalDepth1(self):
    from_node = (0, 0)
    depth = 1
    children_paths = GetChildrenPaths(self.tree, from_node, depth)
    expected_children_paths = [(0, 0, 0)]
    self.assertItemsEqual(expected_children_paths, children_paths)

  def test_PreterminalDepth2(self):
    from_node = (0, 0)
    depth = 2
    children_paths = GetChildrenPaths(self.tree, from_node, depth)
    expected_children_paths = [(0, 0, 0)]
    self.assertItemsEqual(expected_children_paths, children_paths)

  def test_NonterminalDepth1(self):
    from_node = (0,)
    depth = 1
    children_paths = GetChildrenPaths(self.tree, from_node, depth)
    expected_children_paths = [(0, 0), (0, 1)]
    self.assertItemsEqual(expected_children_paths, children_paths)

  def test_NonterminalDepth2(self):
    from_node = (0,)
    depth = 2
    children_paths = GetChildrenPaths(self.tree, from_node, depth)
    expected_children_paths = [(0, 0), (0, 0, 0), (0, 1), (0, 1, 0)]
    self.assertItemsEqual(expected_children_paths, children_paths)

class GetInnerNodesTestCase(unittest.TestCase):
  def setUp(self):
    self.tree = Tree.fromstring('(S (NP (DT the) (NN house)) (VP (VBZ is) (JJ beautiful)))')

  def test_Terminal(self):
    from_node = (0, 0, 0)
    inner_nodes = self.tree.GetInnerNodes(from_node)
    expected_inner_nodes = []
    self.assertItemsEqual(expected_inner_nodes, inner_nodes)

  def test_Preterminal(self):
    from_node = (0, 0)
    inner_nodes = self.tree.GetInnerNodes(from_node)
    expected_inner_nodes = ['DT']
    self.assertItemsEqual(expected_inner_nodes, inner_nodes)

  def test_Nonterminal(self):
    from_node = (0,)
    inner_nodes = self.tree.GetInnerNodes(from_node)
    expected_inner_nodes = ['NP', 'DT', 'NN']
    self.assertItemsEqual(expected_inner_nodes, inner_nodes)

  def test_Root(self):
    from_node = ()
    inner_nodes = self.tree.GetInnerNodes(from_node)
    expected_inner_nodes = ['S', 'NP', 'DT', 'NN', 'VP', 'VBZ', 'JJ']
    self.assertItemsEqual(expected_inner_nodes, inner_nodes)

class GetPathsSpanningLeavesTestCase(unittest.TestCase):
  def setUp(self):
    self.tree = Tree.fromstring('(S (NP (DT the) (NN house)) (VP (VBZ is) (JJ beautiful)))')

  def test_SpanLength1Index0(self):
    root = ()
    paths = self.tree.GetPathsSpanningLeaves(0, 0)
    expected_paths = [(0, 0), (0, 0, 0)]
    self.assertItemsEqual(expected_paths, paths)

  def test_SpanLength1Index2(self):
    root = ()
    paths = self.tree.GetPathsSpanningLeaves(3, 3)
    expected_paths = [(1, 1), (1, 1, 0)]
    self.assertItemsEqual(expected_paths, paths)

  def test_SpanLength2Constituent(self):
    root = ()
    paths = self.tree.GetPathsSpanningLeaves(0, 1)
    expected_paths = [(0,), (0, 0), (0, 0, 0), (0, 1), (0, 1, 0)]
    self.assertItemsEqual(expected_paths, paths)

  def test_SpanLength2NotConstituent(self):
    root = ()
    paths = self.tree.GetPathsSpanningLeaves(1, 2)
    expected_paths = [(0, 1), (0, 1, 0), (1, 0), (1, 0, 0)]
    self.assertItemsEqual(expected_paths, paths)

  def test_SpanFull(self):
    root = ()
    paths = self.tree.GetPathsSpanningLeaves(0, 3)
    expected_paths = self.tree.treepositions()
    self.assertItemsEqual(expected_paths, paths)

  def test_SpanFullRight(self):
    root = (1,)
    paths = self.tree.GetPathsSpanningLeaves(2, 3)
    expected_paths = [(1,), (1, 0), (1, 0, 0), (1, 1), (1, 1, 0)]
    self.assertItemsEqual(expected_paths, paths)

class TreePatternTestCase(unittest.TestCase):
  def setUp(self):
    self.tree = Tree.fromstring('(S (NP (DT the) (NN house)) (VP (VBZ is) (JJ beautiful)))')

  def test_IsStringPreterminal(self):
    path = (0,0,)
    subpaths = []
    tree_pattern = TreePattern(self.tree, path, subpaths)
    self.assertFalse(tree_pattern.IsString())

  def test_IsStringPreterminalWithVar(self):
    path = (0,0,)
    subpaths = [(0,0,0)]
    tree_pattern = TreePattern(self.tree, path, subpaths)
    self.assertFalse(tree_pattern.IsString())

  def test_IsStringTerminal(self):
    path = (0,0,0)
    subpaths = []
    tree_pattern = TreePattern(self.tree, path, subpaths)
    self.assertTrue(tree_pattern.IsString())

  def test_IsStringTerminalVariable(self):
    path = (0,0,0)
    subpaths = [(0,0,0)]
    tree_pattern = TreePattern(self.tree, path, subpaths)
    self.assertFalse(tree_pattern.IsString())

  def test_GetInnerNodesTerminal(self):
    path = (0,0,0)
    subpaths = []
    tree_pattern = TreePattern(self.tree, path, subpaths)
    inner_nodes = tree_pattern.GetInnerNodes()
    expected_inner_nodes = []
    self.assertEqual(expected_inner_nodes, inner_nodes)

  def test_GetInnerNodesPreterminal(self):
    path = (0,1)
    subpaths = []
    tree_pattern = TreePattern(self.tree, path, subpaths)
    inner_nodes = tree_pattern.GetInnerNodes()
    expected_inner_nodes = ['NN']
    self.assertEqual(expected_inner_nodes, inner_nodes)

  def test_GetInnerNodesPreterminalSubpath(self):
    path = (0,1)
    subpaths = [(0,1)]
    tree_pattern = TreePattern(self.tree, path, subpaths)
    inner_nodes = tree_pattern.GetInnerNodes()
    expected_inner_nodes = []
    self.assertEqual(expected_inner_nodes, inner_nodes)

  def test_GetInnerNodesNonterminal(self):
    path = (1,)
    subpaths = []
    tree_pattern = TreePattern(self.tree, path, subpaths)
    inner_nodes = tree_pattern.GetInnerNodes()
    expected_inner_nodes = ['VP', 'VBZ', 'JJ']
    self.assertEqual(expected_inner_nodes, inner_nodes)

  def test_GetInnerNodesNonterminalSubpath0(self):
    path = (1,)
    subpaths = [(1,0)]
    tree_pattern = TreePattern(self.tree, path, subpaths)
    inner_nodes = tree_pattern.GetInnerNodes()
    expected_inner_nodes = ['VP', 'JJ']
    self.assertEqual(expected_inner_nodes, inner_nodes)

  def test_GetInnerNodesNonterminalSubpath00(self):
    path = (1,)
    subpaths = [(1,0,0)]
    tree_pattern = TreePattern(self.tree, path, subpaths)
    inner_nodes = tree_pattern.GetInnerNodes()
    expected_inner_nodes = ['VP', 'VBZ', 'JJ']
    self.assertEqual(expected_inner_nodes, inner_nodes)

  def test_GetInnerNodesNonterminalSubpaths(self):
    path = (1,)
    subpaths = [(1,0), (1,1)]
    tree_pattern = TreePattern(self.tree, path, subpaths)
    inner_nodes = tree_pattern.GetInnerNodes()
    expected_inner_nodes = ['VP']
    self.assertEqual(expected_inner_nodes, inner_nodes)

  def test_GetNumNodesForTerminal(self):
    path = (0,0,0)
    subpaths = []
    tree_pattern = TreePattern(self.tree, path, subpaths)
    num_nodes = tree_pattern.GetNumNodes()
    expected_num_nodes = 1
    self.assertEqual(expected_num_nodes, num_nodes)

  def test_GetNumNodesForTerminalVariable(self):
    path = (0,0,0)
    subpaths = [(0,0,0)]
    tree_pattern = TreePattern(self.tree, path, subpaths)
    num_nodes = tree_pattern.GetNumNodes()
    expected_num_nodes = 0
    self.assertEqual(expected_num_nodes, num_nodes)

  def test_GetNumNodesForPreterminal(self):
    path = (0,0)
    subpaths = []
    tree_pattern = TreePattern(self.tree, path, subpaths)
    num_nodes = tree_pattern.GetNumNodes()
    expected_num_nodes = 2
    self.assertEqual(expected_num_nodes, num_nodes)

  def test_GetNumNodesForPreterminalVariable(self):
    path = (0,0)
    subpaths = [(0,0,0)]
    tree_pattern = TreePattern(self.tree, path, subpaths)
    num_nodes = tree_pattern.GetNumNodes()
    expected_num_nodes = 1
    self.assertEqual(expected_num_nodes, num_nodes)

  def test_GetNumNodesForNonterminal(self):
    path = (0,)
    subpaths = []
    tree_pattern = TreePattern(self.tree, path, subpaths)
    num_nodes = tree_pattern.GetNumNodes()
    expected_num_nodes = 5
    self.assertEqual(expected_num_nodes, num_nodes)

  def test_GetNumNodesForNonterminalVariables(self):
    path = (0,)
    subpaths = [(0,0), (0,1)]
    tree_pattern = TreePattern(self.tree, path, subpaths)
    num_nodes = tree_pattern.GetNumNodes()
    expected_num_nodes = 1
    self.assertEqual(expected_num_nodes, num_nodes)

  def test_GetNumNodesForNonterminalOneVariable(self):
    path = (0,)
    subpaths = [(0,0)]
    tree_pattern = TreePattern(self.tree, path, subpaths)
    num_nodes = tree_pattern.GetNumNodes()
    expected_num_nodes = 3
    self.assertEqual(expected_num_nodes, num_nodes)

  def test_GetNumSubtreesTerminal(self):
    path = (0,0,0)
    subpaths = []
    tree_pattern = TreePattern(self.tree, path, subpaths)
    num_subtrees = tree_pattern.GetNumSubtrees()
    expected_num_subtrees = 0
    self.assertEqual(expected_num_subtrees, num_subtrees)

  def test_GetNumSubtreesPreterminal(self):
    path = (0,0)
    subpaths = []
    tree_pattern = TreePattern(self.tree, path, subpaths)
    num_subtrees = tree_pattern.GetNumSubtrees()
    expected_num_subtrees = 1
    self.assertEqual(expected_num_subtrees, num_subtrees)

  def test_GetNumSubtreesPreterminalWithVar(self):
    path = (0,0)
    subpaths = [(0,0,0)]
    tree_pattern = TreePattern(self.tree, path, subpaths)
    num_subtrees = tree_pattern.GetNumSubtrees()
    expected_num_subtrees = 1
    self.assertEqual(expected_num_subtrees, num_subtrees)

  def test_GetLeavesButNoLeavesPathRootSubpathsFull(self):
    path = ()
    subpaths = [(0,), (1,)]
    tree_pattern = TreePattern(self.tree, path, subpaths)
    leaves = tree_pattern.GetLeaves()
    expected_leaves = []
    self.assertListEqual(expected_leaves, leaves)
    leaves_indices = tree_pattern.GetLeavesIndices()
    expected_indices = []
    self.assertListEqual(expected_indices, leaves_indices)

  def test_GetLeavesButNoLeavesPathRootSubpathsLeft(self):
    path = ()
    subpaths = [(0,)]
    tree_pattern = TreePattern(self.tree, path, subpaths)
    leaves = tree_pattern.GetLeaves()
    expected_leaves = ['is', 'beautiful']
    self.assertListEqual(expected_leaves, leaves)
    leaves_indices = tree_pattern.GetLeavesIndices()
    expected_indices = [2, 3]
    self.assertListEqual(expected_indices, leaves_indices)

  def test_GetLeavesButNoLeavesPathRootSubpathsRight(self):
    path = ()
    subpaths = [(1,)]
    tree_pattern = TreePattern(self.tree, path, subpaths)
    leaves = tree_pattern.GetLeaves()
    expected_leaves = ['the', 'house']
    self.assertListEqual(expected_leaves, leaves)
    leaves_indices = tree_pattern.GetLeavesIndices()
    expected_indices = [0, 1]
    self.assertListEqual(expected_indices, leaves_indices)

  def test_GetLeavesButNoLeavesPathRootSubpathsLeft2(self):
    path = ()
    subpaths = [(0,1)]
    tree_pattern = TreePattern(self.tree, path, subpaths)
    leaves = tree_pattern.GetLeaves()
    expected_leaves = ['the', 'is', 'beautiful']
    self.assertListEqual(expected_leaves, leaves)
    leaves_indices = tree_pattern.GetLeavesIndices()
    expected_indices = [0, 2, 3]
    self.assertListEqual(expected_indices, leaves_indices)

  def test_GetLeavesButNoLeavesPath0SubpathsLeft2(self):
    path = (0,)
    subpaths = [(0,1)]
    tree_pattern = TreePattern(self.tree, path, subpaths)
    leaves = tree_pattern.GetLeaves()
    expected_leaves = ['the']
    self.assertListEqual(expected_leaves, leaves)
    leaves_indices = tree_pattern.GetLeavesIndices()
    expected_indices = [0]
    self.assertListEqual(expected_indices, leaves_indices)

  def test_GetLeavesButNoLeavesPath00Subpaths00(self):
    path = (0,0)
    subpaths = [(0,0)]
    tree_pattern = TreePattern(self.tree, path, subpaths)
    leaves = tree_pattern.GetLeaves()
    expected_leaves = []
    self.assertListEqual(expected_leaves, leaves)
    leaves_indices = tree_pattern.GetLeavesIndices()
    expected_indices = []
    self.assertListEqual(expected_indices, leaves_indices)

  def test_GetLeavesPathLeafNoSubpaths(self):
    path = (0,0)
    subpaths = []
    tree_pattern = TreePattern(self.tree, path, subpaths)
    leaves = tree_pattern.GetLeaves()
    expected_leaves = ['the']
    self.assertListEqual(expected_leaves, leaves)
    leaves_indices = tree_pattern.GetLeavesIndices()
    expected_indices = [0]
    self.assertListEqual(expected_indices, leaves_indices)

  def test_GetLeavesFromStringTreePattern(self):
    tree = 'the'
    path = ()
    subpaths = [()]
    tree_pattern = TreePattern(tree, path, subpaths)
    leaves = tree_pattern.GetLeaves()
    expected_leaves = []
    self.assertListEqual(expected_leaves, leaves)
    leaves_indices = tree_pattern.GetLeavesIndices()
    expected_indices = []
    self.assertListEqual(expected_indices, leaves_indices)

  def test_GetLeavesFromStringTreePattern1(self):
    tree = 'the'
    path = ()
    subpaths = []
    tree_pattern = TreePattern(tree, path, subpaths)
    leaves = tree_pattern.GetLeaves()
    expected_leaves = ['the']
    self.assertListEqual(expected_leaves, leaves)
    leaves_indices = tree_pattern.GetLeavesIndices()
    expected_indices = [0]
    self.assertListEqual(expected_indices, leaves_indices)

  def test_GetLeavesInTreePatternRepeatedLeaves1(self):
    tree = tree_or_string('(A (B b) (C b))')
    path = ()
    subpaths = [(0,)]
    tree_pattern = TreePattern(tree, path, subpaths)
    leaves = tree_pattern.GetLeaves()
    expected_leaves = ['b']
    self.assertListEqual(expected_leaves, leaves)
    leaves_indices = tree_pattern.GetLeavesIndices()
    expected_indices = [1]
    self.assertListEqual(expected_indices, leaves_indices)

  def test_GetLeavesInTreePatternRepeatedLeaves2(self):
    tree = tree_or_string('(A (B b) (C b))')
    path = ()
    subpaths = [(1,)]
    tree_pattern = TreePattern(tree, path, subpaths)
    leaves = tree_pattern.GetLeaves()
    expected_leaves = ['b']
    self.assertListEqual(expected_leaves, leaves)
    leaves_indices = tree_pattern.GetLeavesIndices()
    expected_indices = [0]
    self.assertListEqual(expected_indices, leaves_indices)

  def test_GetNodesFromLeaf(self):
    path = (0,0,0)
    subpaths = []
    tree_pattern = TreePattern(self.tree, path, subpaths)
    nodes = tree_pattern.GetNodes()
    expected_nodes = ['the']
    self.assertListEqual(expected_nodes, nodes)

  def test_GetNodesFromPreterminal(self):
    path = (0,0)
    subpaths = []
    tree_pattern = TreePattern(self.tree, path, subpaths)
    nodes = tree_pattern.GetNodes()
    expected_nodes = ['DT', 'the']
    self.assertListEqual(expected_nodes, nodes)

  def test_GetNodesFromNonterminal(self):
    path = (0,)
    subpaths = []
    tree_pattern = TreePattern(self.tree, path, subpaths)
    nodes = tree_pattern.GetNodes()
    expected_nodes = ['NP', 'DT', 'the', 'NN', 'house']
    self.assertListEqual(expected_nodes, nodes)

  def test_GetNodesFromPreterminalSubpathLeaf(self):
    path = (0,0)
    subpaths = [(0,0,0)]
    tree_pattern = TreePattern(self.tree, path, subpaths)
    nodes = tree_pattern.GetNodes()
    expected_nodes = ['DT']
    self.assertListEqual(expected_nodes, nodes)

  def test_GetNodesFromNonterminalSubpathLeaf(self):
    path = (0,)
    subpaths = [(0,0,0)]
    tree_pattern = TreePattern(self.tree, path, subpaths)
    nodes = tree_pattern.GetNodes()
    expected_nodes = ['NP', 'DT', 'NN', 'house']
    self.assertListEqual(expected_nodes, nodes)

  def test_GetNodesFromNonterminalSubpathLeaves(self):
    path = (0,)
    subpaths = [(0,0,0), (0,1,0)]
    tree_pattern = TreePattern(self.tree, path, subpaths)
    nodes = tree_pattern.GetNodes()
    expected_nodes = ['NP', 'DT', 'NN']
    self.assertListEqual(expected_nodes, nodes)

  def test_GetNodesFromNonterminalSubpathLeafAndPreterminal(self):
    path = (0,)
    subpaths = [(0,0,0), (0,1)]
    tree_pattern = TreePattern(self.tree, path, subpaths)
    nodes = tree_pattern.GetNodes()
    expected_nodes = ['NP', 'DT']
    self.assertListEqual(expected_nodes, nodes)

  def test_GetNodesNoLeavesLeft(self):
    path = (0,)
    subpaths = []
    tree_pattern = TreePattern(self.tree, path, subpaths)
    nodes = tree_pattern.GetNodes(only_inner=True)
    expected_nodes = ['NP', 'DT', 'NN']
    self.assertListEqual(expected_nodes, nodes)

  def test_GetNodesNoLeavesLeftRight(self):
    path = (0,1)
    subpaths = []
    tree_pattern = TreePattern(self.tree, path, subpaths)
    nodes = tree_pattern.GetNodes(only_inner=True)
    expected_nodes = ['NN']
    self.assertListEqual(expected_nodes, nodes)

  def test_GetNodesNoLeavesLeftRightLeft(self):
    path = (0,1,0)
    subpaths = []
    tree_pattern = TreePattern(self.tree, path, subpaths)
    nodes = tree_pattern.GetNodes(only_inner=True)
    expected_nodes = []
    self.assertListEqual(expected_nodes, nodes)

class GetPosAtTestCase(unittest.TestCase):
  def test_Terminal(self):
    tree = tree_or_string('the')
    pos = GetPosAt(tree, ())
    self.assertEqual('the', pos)

  def test_TerminalBadPath(self):
    tree = tree_or_string('the')
    pos = GetPosAt(tree, (0,))
    self.assertEqual('the', pos)

  def test_UntypedVar(self):
    tree = tree_or_string('?x0|')
    pos = GetPosAt(tree, ())
    self.assertEqual('', pos)

  def test_TypedVar(self):
    tree = tree_or_string('?x0|NP')
    pos = GetPosAt(tree, ())
    self.assertEqual('NP', pos)

  def test_TypedVar2Level(self):
    tree = tree_or_string('(NP ?x0|DT)')
    pos = GetPosAt(tree, (0,))
    self.assertEqual('DT', pos)
    pos = GetPosAt(tree, ())
    self.assertEqual('NP', pos)

if __name__ == '__main__':
  suite1  = unittest.TestLoader().loadTestsFromTestCase(TreePatternTestCase)
  suite2  = unittest.TestLoader().loadTestsFromTestCase(GetPathsSpanningLeavesTestCase)
  suite3  = unittest.TestLoader().loadTestsFromTestCase(GetInnerNodesTestCase)
  suite4  = unittest.TestLoader().loadTestsFromTestCase(GetChildrenPathsTestCase)
  # suite2  = unittest.TestLoader().loadTestsFromTestCase(NumSubtreesTestCase)
  suite5  = unittest.TestLoader().loadTestsFromTestCase(TreeContainsTestCase)
  suite6  = unittest.TestLoader().loadTestsFromTestCase(GetPosAtTestCase)
  suite7  = unittest.TestLoader().loadTestsFromTestCase(ToXmlTestCase)
  suite8  = unittest.TestLoader().loadTestsFromTestCase(IsPlausibleEntityPhraseTestCase)
  suites  = unittest.TestSuite([suite1, suite2, suite3, suite4, suite5, suite6, suite7, suite8])
  unittest.TextTestRunner(verbosity=2).run(suites)

