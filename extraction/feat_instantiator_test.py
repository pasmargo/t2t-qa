# -*- coding: utf-8 -*-
import unittest

import math
import os

from extraction.extractor_beam import Transformation
from extraction.feat_instantiator import FeatureInstantiator
from training.transductionrule import XTRule
from utils.tree_tools import TreePattern, tree_or_string

FeatureInstantiator.feature_templates = [
  'roots',
  # 'identity',
  # 'yield_words',
  'num_variables',
  'num_del_variables',
  'num_leaves',
  'num_leaves_diff',
  'uri_role',
  'state',
  'newstates',
  'surf_to_uri_info',
  'surface_overlap',
  'trg_surface_overlap',
  'trg_surface_overlap_to_context',
  'fb_typed_size',
  'tf_idf',
  'count_op',
  'src_context_to_uri',
  'undeleted_how_many',
  'type_match']

class LexicalFeaturesTestCase(unittest.TestCase):
  def setUp(self):
    self.src_tree = tree_or_string(
      '(S (NP (DT the) (JJ nice) (NN house)) (VP (VBZ is) (JJ small)))')
    self.trg_tree = tree_or_string(
      '(S (NP (DT la) (NN casa) (JJ bonita)) (VP (VBZ es) (JJ pequena)))')
    self.description_file = 'extraction/feature_description_test.txt'
    self.feat_instantiator = FeatureInstantiator(
      self.description_file, feats_cache_filename='extraction/.feats_cache_test')

  def tearDown(self):
    self.feat_instantiator.Close()
    self.feat_instantiator = None
    # os.unlink(self.description_file)

  def AssertFeaturesInByName(
      self, rule_features, expected_features, feat_instantiator, feat_in=True):
    self.assertFalse(rule_features is None and len(expected_features) > 0)
    if rule_features is None:
      return
    rule_feat_ids = [f[0] for f in rule_features]
    for feature_name, feat_val in expected_features:
      feature_id = feat_instantiator.GetFeatureIDByName(feature_name)
      dummy_rule = XTRule('q', tree_or_string('a'), tree_or_string('b'), {}, 1.0)
      dummy_rule.features = rule_features
      if feat_in:
        self.assertIn(feature_id, rule_feat_ids,
                      '{0} with ID {1} not found in:\n{2}'\
        .format(feature_name, feature_id,
                feat_instantiator.DescribeFeatureIDs(dummy_rule)))
        feat_index = rule_feat_ids.index(feature_id)
        self.assertAlmostEqual(feat_val, rule_features[feat_index][-1], 3,
                      msg='{0}\'s value {3} with ID {1} not found in:\n{2}'\
        .format(feature_name, feature_id,
                feat_instantiator.DescribeFeatureIDs(dummy_rule), feat_val))
                      
      else:
        self.assertNotIn(feature_id, rule_feat_ids,
                         '{0} with ID {1} found in:\n{2}'\
        .format(feature_name, feature_id,
                feat_instantiator.DescribeFeatureIDs(dummy_rule)))
    
  @unittest.skipIf('roots' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_RootsRule(self):
    src_path = (0,)
    src_subpaths = [(0, 0,), (0, 1,), (0, 2)]
    src_treep = TreePattern(self.src_tree, src_path, src_subpaths)
    trg_path = (1, 0)
    trg_subpaths = [(1, 0, 0)]
    trg_treep = TreePattern(self.trg_tree, trg_path, trg_subpaths)
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep)
    expected_features = [
      (('root: lhs', 'NP'), 1.0),
      (('root: rhs', 'VBZ'), 1.0),
      (('root: lhs, rhs', ('NP', 'VBZ')), 1.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)

  @unittest.skipIf('roots' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_RootsRuleVar(self):
    src_path = (0,)
    src_subpaths = [(0,)]
    src_treep = TreePattern(self.src_tree, src_path, src_subpaths)
    trg_path = (1, 0)
    trg_subpaths = [(1, 0, 0)]
    trg_treep = TreePattern(self.trg_tree, trg_path, trg_subpaths)
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep)
    expected_features = [
      (('root: lhs', ''), 1.0),
      (('root: rhs', 'VBZ'), 1.0),
      (('root: lhs, rhs', ('', 'VBZ')), 1.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)

  @unittest.skipIf('identity' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_Identity(self):
    src_path = (1,)
    src_subpaths = [(1, 0,), (1, 1,)]
    src_treep = TreePattern(self.src_tree, src_path, src_subpaths)
    trg_path = (1,)
    trg_subpaths = [(1, 0,), (1, 1,)]
    trg_treep = TreePattern(self.trg_tree, trg_path, trg_subpaths)
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep)
    expected_features = [
      (('identity: lhs', '(VP ?x0|VBZ ?x1|JJ)'), 1.0),
      (('identity: rhs', '(VP ?x0|VBZ ?x1|JJ)'), 1.0),
      (('identity: lhs, rhs', ('(VP ?x0|VBZ ?x1|JJ)', '(VP ?x0|VBZ ?x1|JJ)')), 1.0),
      ('identity: lhs and rhs are equal', 1.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)

  @unittest.skipIf('yield_words' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_YieldSingle(self):
    src_path = (1,)
    src_subpaths = [(1, 0,)]
    src_treep = TreePattern(self.src_tree, src_path, src_subpaths)
    trg_path = (1,)
    trg_subpaths = [(1, 0,)]
    trg_treep = TreePattern(self.trg_tree, trg_path, trg_subpaths)
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep)
    expected_features = [
      (('yield: lhs', ('small',)), 1.0),
      (('yield: rhs', ('pequena',)), 1.0),
      (('yield: lhs, rhs', (('small',), ('pequena',))), 1.0),
      (('yield: common in lhs and rhs', ()), 1.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)

  @unittest.skipIf('yield_words' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_YieldSingleVar(self):
    src_path = (1,)
    src_subpaths = [(1,)]
    src_treep = TreePattern(self.src_tree, src_path, src_subpaths)
    trg_path = (1,)
    trg_subpaths = [(1, 0,)]
    trg_treep = TreePattern(self.trg_tree, trg_path, trg_subpaths)
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep)
    expected_features = [
      (('yield: lhs', ()), 1.0),
      (('yield: rhs', ('pequena',)), 1.0),
      (('yield: lhs, rhs', ((), ('pequena',))), 1.0),
      (('yield: common in lhs and rhs', ()), 1.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)

  @unittest.skipIf('yield_words' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_YieldMultiple(self):
    src_path = (0,)
    src_subpaths = [(0, 0,)]
    src_treep = TreePattern(self.src_tree, src_path, src_subpaths)
    trg_path = (0,)
    trg_subpaths = []
    trg_treep = TreePattern(self.trg_tree, trg_path, trg_subpaths)
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep)
    expected_features = [
      (('yield: lhs', ('nice', 'house')), 1.0),
      (('yield: rhs', ('la', 'casa', 'bonita')), 1.0),
      (('yield: lhs, rhs', (('nice', 'house'), ('la', 'casa', 'bonita'))), 1.0),
      (('yield: common in lhs and rhs', ()), 1.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)

  @unittest.skipIf('num_variables' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_NumVariables(self):
    src_path = (1,)
    src_subpaths = [(1, 0,), (1, 1)]
    src_treep = TreePattern(self.src_tree, src_path, src_subpaths)
    trg_path = (1,)
    trg_subpaths = [(1, 0,)]
    trg_treep = TreePattern(self.trg_tree, trg_path, trg_subpaths)
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep)
    expected_features = [
      (('num_variables: lhs', 2), 1.0),
      (('num_variables: rhs', 1), 1.0),
      (('num_variables: abs(lhs - rhs)', 1), 1.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)

  @unittest.skipIf('num_variables' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_NumVariables2(self):
    rule = XTRule('q1',
                  tree_or_string('?x0|NP'),
                  tree_or_string('(NP ?x0|)'),
                  {(0, ) : 'q2'}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep)
    expected_features = [
      (('num_variables: lhs', 1), 1.0),
      (('num_variables: rhs', 1), 1.0),
      (('num_variables: abs(lhs - rhs)', 0), 1.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)

  @unittest.skipIf('num_variables' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_NumVariablesDeletion(self):
    rule = XTRule('q1',
                  tree_or_string('(NP ?xx0|DT ?x0|NN)'),
                  tree_or_string('(NP ?x0|)'),
                  {(0, ) : 'q2'}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep)
    expected_features = [
      (('num_variables: lhs', 2), 1.0),
      (('num_variables: rhs', 1), 1.0),
      (('num_variables: abs(lhs - rhs)', 1), 1.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)

  @unittest.skipIf('num_del_variables' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_NumDelVariables(self):
    rule = XTRule('q1',
                  tree_or_string('(NP ?xx0|DT ?x0|NN ?xx1|JJ)'),
                  tree_or_string('(NP ?x0|)'),
                  {(0, ) : 'q2'}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep)
    expected_features = [
      (('num_del_variables: lhs', 2), 1.0),
      (('num_del_variables: rhs', 0), 1.0),
      (('num_del_variables: abs(lhs - rhs)', 2), 1.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)

  @unittest.skipIf('num_leaves' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_NumLeaves(self):
    src_path = (1,)
    src_subpaths = [(1, 0,), (1, 1)]
    src_treep = TreePattern(self.src_tree, src_path, src_subpaths)
    trg_path = (1,)
    trg_subpaths = [(1, 0,)]
    trg_treep = TreePattern(self.trg_tree, trg_path, trg_subpaths)
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep)
    expected_features = [
      (('num_leaves: lhs', 0), 1.0),
      (('num_leaves: rhs', 1), 1.0),
      (('num_leaves: abs(lhs - rhs)', 1), 1.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)

  @unittest.skipIf('num_leaves' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_NumLeaves2(self):
    src_path = (0, 0, 0)
    src_subpaths = []
    src_treep = TreePattern(self.src_tree, src_path, src_subpaths)
    trg_path = (0,)
    trg_subpaths = [(0, 0)]
    trg_treep = TreePattern(self.trg_tree, trg_path, trg_subpaths)
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep)
    expected_features = [
      (('num_leaves: lhs', 1), 1.0),
      (('num_leaves: rhs', 2), 1.0),
      (('num_leaves: abs(lhs - rhs)', 1), 1.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)

  @unittest.skipIf('num_leaves' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_NumLeaves3(self):
    src_path = (0,)
    src_subpaths = [(0, 0,), (0, 1,), (0, 2)]
    src_treep = TreePattern(self.src_tree, src_path, src_subpaths)
    trg_path = (0,)
    trg_subpaths = [(0, 0,), (0, 2)]
    trg_treep = TreePattern(self.trg_tree, trg_path, trg_subpaths)
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep)
    expected_features = [
      (('num_leaves: lhs', 0), 1.0),
      (('num_leaves: rhs', 1), 1.0),
      (('num_leaves: abs(lhs - rhs)', 1), 1.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)

  @unittest.skipIf('num_leaves_diff' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_NumLeavesDiff(self):
    src_path = (0,)
    src_subpaths = []
    src_treep = TreePattern(self.src_tree, src_path, src_subpaths)
    trg_path = (0,)
    trg_subpaths = [(0, 0,), (0, 1), (0, 2)]
    trg_treep = TreePattern(self.trg_tree, trg_path, trg_subpaths)
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep)
    expected_features = [(('num_leaves_diff: abs(lhs - rhs) >=', 3), 1.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)

  @unittest.skipIf('num_leaves_diff' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_NumLeavesDiff2(self):
    src_path = ()
    src_subpaths = []
    src_treep = TreePattern(self.src_tree, src_path, src_subpaths)
    trg_path = (0,)
    trg_subpaths = [(0, 0,), (0, 1), (0, 2)]
    trg_treep = TreePattern(self.trg_tree, trg_path, trg_subpaths)
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep)
    expected_features = [(('num_leaves_diff: abs(lhs - rhs) >=', 3), 1.0),
                         (('num_leaves_diff: abs(lhs - rhs) >=', 5), 1.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)

  @unittest.skipIf('uri_role' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_URIRole(self):
    # Source tree pattern is irrelevant here.
    src_path = (0, 0, 0)
    src_subpaths = []
    src_treep = TreePattern(self.src_tree, src_path, src_subpaths)
    trg_tree = tree_or_string(
      '(ID !fb:education.academic_post.institution '
          '(ID fb:education.academic_post.person fb:en.marshall_hall) '
          '(ID fb:education.academic_post.position_or_title fb:en.professor))')
    trg_path = (0,)
    trg_subpaths = []
    trg_treep = TreePattern(trg_tree, trg_path, trg_subpaths)
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep)
    expected_roles = ('predicate',)
    expected_features = [(('uri_role: rhs', expected_roles), 1.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)

  @unittest.skipIf('uri_role' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_URIRole2(self):
    # Source tree pattern is irrelevant here.
    src_path = (0, 0, 0)
    src_subpaths = []
    src_treep = TreePattern(self.src_tree, src_path, src_subpaths)
    trg_tree = tree_or_string(
      '(ID !fb:education.academic_post.institution '
          '(ID fb:education.academic_post.person fb:en.marshall_hall) '
          '(ID fb:education.academic_post.position_or_title fb:en.professor))')
    trg_path = (1,)
    trg_subpaths = []
    trg_treep = TreePattern(trg_tree, trg_path, trg_subpaths)
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep)
    expected_roles = ('predicate', 'entity')
    expected_features = [(('uri_role: rhs', expected_roles), 1.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)

  @unittest.skipIf('uri_role' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_URIRoleDate(self):
    src_path = (0, 0, 0)
    src_subpaths = []
    src_treep = TreePattern(self.src_tree, src_path, src_subpaths)
    trg_tree = tree_or_string(
      '(ID !fb:education.academic_post.institution '
          '(DATE 20_1_2015) '
          '(ID fb:education.academic_post.position_or_title fb:en.professor))')
    trg_path = (1,)
    trg_subpaths = []
    trg_treep = TreePattern(trg_tree, trg_path, trg_subpaths)
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep)
    expected_roles = ()
    expected_features = [(('uri_role: rhs', expected_roles), 1.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)

  @unittest.skipIf('state' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_State(self):
    rule = XTRule('q1',
                  tree_or_string('(NP ?x0|DT ?x1|NN ?xx1|JJ)'),
                  tree_or_string('(NP ?x0|DT ?x1|NN)'),
                  {(0,) : 'q3', (1,) : 'q2'}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    expected_features = [(('state', 'q1'), 1.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)

  @unittest.skipIf('newstates' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_Newstates(self):
    rule = XTRule('q1',
                  tree_or_string('(NP ?x0|DT ?x1|NN ?xx1|JJ)'),
                  tree_or_string('(NP ?x0|DT ?x1|NN)'),
                  {(0,) : 'q3', (1,) : 'q2'}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    expected_features = [(('newstates', ('q3', 'q2')), 1.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)

  @unittest.skipIf('newstates' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_NewstatesEmpty(self):
    rule = XTRule('q1',
                  tree_or_string('(NP ?x0|DT ?x1|NN ?xx1|JJ)'),
                  tree_or_string('(DT the)'),
                  {}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    expected_features = [(('newstates', ()), 1.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)

  @unittest.skipIf('surface_overlap' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_SurfaceMatch(self):
    rule = XTRule('q1',
                  tree_or_string('professor'),
                  tree_or_string('fb:en.professor'),
                  {}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    # expected_features = [(('surface_match', '80<=ratio'), 1.0)]
    expected_features = [('surface_match', 1.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)

  @unittest.skipIf('surface_overlap' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_SurfaceMatchNP(self):
    rule = XTRule('q1',
                  tree_or_string('(NP (DT a) (NN professor))'),
                  tree_or_string('fb:en.professor'),
                  {}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    # expected_features = [(('surface_match', '80<=ratio'), 1.0)]
    expected_features = [('surface_match', 1.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)

  @unittest.skipIf('surface_overlap' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_SurfaceLowMatchNPLong(self):
    rule = XTRule('q1',
                  tree_or_string('(NP (DT a) (JJ distinguished) (NN professor))'),
                  tree_or_string('fb:en.professor'),
                  {}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    # 3-grams:
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    # expected_features = [(('surface_match', '20<=ratio<40'), 1.0)]
    expected_features = [('surface_match', 0.3888888)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)
    # 2-grams:
    self.feat_instantiator.n_grams = 2
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    # expected_features = [(('surface_match', '40<=ratio<60'), 1.0)]
    expected_features = [('surface_match', 0.4)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)

  @unittest.skipIf('surface_overlap' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_SurfaceLowMatch(self):
    rule = XTRule('q1',
                  tree_or_string('fb:en.marshall_hall'),
                  tree_or_string('fb:en.professor'),
                  {}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    # expected_features = [(('surface_match', '0<=ratio<20'), 1.0)]
    expected_features = [('surface_match', 0.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)

  @unittest.skipIf('surface_overlap' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_SurfaceHighMatchLongURI(self):
    rule = XTRule('q1',
                  tree_or_string('(NP (NNP institutions))'),
                  tree_or_string('!fb:education.academic_post.institution'),
                  {}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    # expected_features = [(('surface_match', '80<=ratio'), 1.0)]
    expected_features = [('surface_match', 0.9)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)

  @unittest.skipIf('surface_overlap' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_SurfaceHighMatchBridgeLongURI(self):
    rule = XTRule('q1',
                  tree_or_string('(NP (NNP marshall) (NNP hall))'),
                  tree_or_string('(ID [] fb:en.marshall_hall)'),
                  {}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    # expected_features = [(('surface_match', '80<=ratio'), 1.0)]
    expected_features = [('surface_match', 1.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)

  @unittest.skipIf('surface_overlap' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_SurfaceVeryShort(self):
    rule = XTRule('q1',
                  tree_or_string('(IN mat)'),
                  tree_or_string('cati'),
                  {}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    # 3-grams
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    # expected_features = [(('surface_match', '0<=ratio<20'), 1.0)]
    expected_features = [('surface_match', 0.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)
    # 2-grams
    self.feat_instantiator.n_grams = 2
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    # expected_features = [(('surface_match', '80<=ratio'), 1.0)]
    expected_features = [('surface_match', .5)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)

  @unittest.skipIf('surface_overlap' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_SurfaceSuperShortNoMatch(self):
    rule = XTRule('q1',
                  tree_or_string('(IN us)'),
                  tree_or_string('uk'),
                  {}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    # 3-grams
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    # expected_features = [(('surface_match', '0<=ratio<20'), 1.0)]
    expected_features = [('surface_match', .0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)
    # 2-grams
    self.feat_instantiator.n_grams = 2
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    # expected_features = [(('surface_match', '80<=ratio'), 1.0)]
    expected_features = [('surface_match', .0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)

  @unittest.skipIf('surface_overlap' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_SurfaceSuperShortMatch(self):
    rule = XTRule('q1',
                  tree_or_string('(IN us)'),
                  tree_or_string('us'),
                  {}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    # 3-grams
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    # expected_features = [(('surface_match', '0<=ratio<20'), 1.0)]
    expected_features = [('surface_match', 1.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)
    # 2-grams
    self.feat_instantiator.n_grams = 2
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    # expected_features = [(('surface_match', '80<=ratio'), 1.0)]
    expected_features = [('surface_match', 1.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)

  @unittest.skipIf('surface_overlap' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_Surface2ShortWords(self):
    rule = XTRule('q1',
                  tree_or_string('(NP (DT the) (NNP nyse))'),
                  tree_or_string('!fb:olympics.olympic_athlete_affiliation.athlete'),
                  {}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    # 3-grams
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    # expected_features = [(('surface_match', '0<=ratio<20'), 1.0)]
    expected_features = [('surface_match', 0.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)
    # 2-grams
    self.feat_instantiator.n_grams = 2
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    # expected_features = [(('surface_match', '0<=ratio<20'), 1.0)]
    expected_features = [('surface_match', 0.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)

  @unittest.skipIf('surface_overlap' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_SurfaceLowMatchLongPhrase(self):
    rule = XTRule('q1',
                  tree_or_string('(WHPP (IN at) (WHNP (WP what) (NP (NNP institutions))))'),
                  tree_or_string('!fb:base.politicalconventions.convention_speech.venue'),
                  {}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    # 3-grams:
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    # expected_features = [(('surface_match', '20<=ratio<40'), 1.0)]
    expected_features = [('surface_match', 0.3)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)
    # 2-grams:
    self.feat_instantiator.n_grams = 2
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    # expected_features = [(('surface_match', '40<=ratio<60'), 1.0)]
    expected_features = [('surface_match', 0.545454545455)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)

  @unittest.skipIf('trg_surface_overlap' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_TrgSurfaceHighMatchShort(self):
    rule = XTRule('q1',
                  tree_or_string('(WHPP (IN in) (WHNP (WP what) (NP (NN area))))'),
                  tree_or_string('!fb:location.location.area'),
                  {}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    # 3-grams:
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    expected_features = [('trg_surface_match', 1.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)
    # 2-grams:
    self.feat_instantiator.n_grams = 2
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    expected_features = [('trg_surface_match', 1.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)

  @unittest.skipIf('trg_surface_overlap' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_TrgSurfaceMediumMatchTwoTokens(self):
    rule = XTRule('q1',
                  tree_or_string('(WHPP (IN in) (WHNP (WP what) (NP (NN area))))'),
                  tree_or_string('!fb:location.location.area_4510'),
                  {}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    # 3-grams:
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    expected_features = [('trg_surface_match', .5)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)
    # 2-grams:
    self.feat_instantiator.n_grams = 2
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    expected_features = [('trg_surface_match', .5)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)

  @unittest.skipIf('trg_surface_overlap_to_context' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_TrgSurfaceToContextHighMatch(self):
    src_tree = tree_or_string('(ROOT (T0 area) (T1 4510))')
    self.feat_instantiator.SetContext({'src_tree' : str(src_tree)})
    rule = XTRule('q1',
                  tree_or_string('(WHPP (IN in) (WHNP (WP what) (NP (NN area))))'),
                  tree_or_string('!fb:location.location.area_4510'),
                  {}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    # 3-grams:
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    expected_features = [('trg_surface_match_to_context', 1.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)
    # 2-grams:
    self.feat_instantiator.n_grams = 2
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    expected_features = [('trg_surface_match_to_context', 1.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)

  @unittest.skipIf('trg_surface_overlap_to_context' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_TrgSurfaceToContextLowMatch(self):
    src_tree = tree_or_string('(ROOT (T0 area))')
    self.feat_instantiator.SetContext({'src_tree' : str(src_tree)})
    rule = XTRule('q1',
                  tree_or_string('(WHPP (IN in) (WHNP (WP what) (NP (NN area))))'),
                  tree_or_string('!fb:location.location.area_4510'),
                  {}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    # 3-grams:
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    expected_features = [('trg_surface_match_to_context', .5)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)
    # 2-grams:
    self.feat_instantiator.n_grams = 2
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    expected_features = [('trg_surface_match_to_context', .5)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)

  @unittest.skipIf('tf_idf' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_TfidfLow(self):
    rule = XTRule('q1',
                  tree_or_string('(NP (DT the) (NNP nyse))'),
                  tree_or_string('!fb:olympics.olympic_athlete_affiliation.athlete'),
                  {}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    expected_features = [('linking_tfidf', 9.0 / 40)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)

  @unittest.skipIf('tf_idf' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_TfidfLow2(self):
    rule = XTRule('q1',
                  tree_or_string('(WHPP (IN at) (WHNP (WP what) (NP (NNP institutions))))'),
                  tree_or_string('!fb:base.politicalconventions.convention_speech.venue'),
                  {}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    expected_features = [('linking_tfidf', 7.0 / 40)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)

  @unittest.skipIf('tf_idf' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_TfidfHigh(self):
    rule = XTRule('q1',
                  tree_or_string('(NP (DT the) (NNP nyse))'),
                  tree_or_string('!fb:en.new_york_stock_exchange_inc'),
                  {}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    expected_features = [('linking_tfidf', 13.0 / 40)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)

  @unittest.skipIf('tf_idf' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_TfidfHigh2(self):
    rule = XTRule('q1',
                  tree_or_string('(NP (DT a) (NN professor))'),
                  tree_or_string('fb:en.professor'),
                  {}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    expected_features = [('linking_tfidf', 12.0 / 40)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)

  @unittest.skipIf('count_op' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_Count(self):
    rule = XTRule('q1',
                  tree_or_string('(ROOT (SBARQ (WHNP (WHADJP (WRB how) (JJ many)) ?x0|NNS) ?x1|SQ (. ?)))'),
                  tree_or_string('(ID COUNT (ID ?x0| ?x1|))'),
                  {(0, 0) : 'q2', (0, 1) : 'q2'}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    expected_features = [(('count_op', True), 1.0)]
    not_expected_features = [(('count_op', False), 1.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)
    self.AssertFeaturesInByName(
      rule_features, not_expected_features, self.feat_instantiator, feat_in=False)

  @unittest.skipIf('count_op' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_CountNo(self):
    rule = XTRule('q1',
                  tree_or_string('(ROOT (SBARQ (WHNP (WHADJP (WRB how) (JJ many)) ?x0|NNS) ?x1|SQ (. ?)))'),
                  tree_or_string('(ID MAX (ID ?x0| ?x1|))'),
                  {(0, 0) : 'q2', (0, 1) : 'q2'}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    expected_features = [(('count_op', False), 1.0)]
    not_expected_features = [(('count_op', True), 1.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator, feat_in=True)
    self.AssertFeaturesInByName(
      rule_features, not_expected_features, self.feat_instantiator, feat_in=False)

  @unittest.skipIf('count_op' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_CountNo2(self):
    rule = XTRule('q1',
                  tree_or_string('(ROOT (SBARQ (WHNP (WHADJP (WRB how) (JJ much)) ?x0|NNS) ?x1|SQ (. ?)))'),
                  tree_or_string('(ID COUNT (ID ?x0| ?x1|))'),
                  {(0, 0) : 'q2', (0, 1) : 'q2'}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    expected_features = [(('count_op', False), 1.0)]
    not_expected_features = [(('count_op', True), 1.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator, feat_in=True)
    self.AssertFeaturesInByName(
      rule_features, not_expected_features, self.feat_instantiator, feat_in=False)

  @unittest.skipIf('count_op' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_CountEmpty(self):
    rule = XTRule('q1',
                  tree_or_string('(ROOT (SBARQ (WHNP (WHADJP (WRB how) (JJ much)) ?x0|NNS) ?x1|SQ (. ?)))'),
                  tree_or_string('(MAX (ID ?x0| ?x1|))'),
                  {(0, 0) : 'q2', (0, 1) : 'q2'}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    not_expected_features = [(('count_op', True), 1.0), (('count_op', False), 1.0)]
    self.AssertFeaturesInByName(
      rule_features, not_expected_features, self.feat_instantiator, feat_in=False)

  @unittest.skipIf('count_op' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_CountPredicateTrue(self):
    rule = XTRule('q1',
                  tree_or_string('(ROOT (SBARQ (WHNP (WHADJP (WRB how) (JJ many)) ?xx0|) ?x1|SQ (. ?)))'),
                  tree_or_string('(ID !fb:architecture.building.floors ?x1|)'),
                  {(1,) : 'q2'}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    expected_features = [(('count_op', True), 1.0)]
    not_expected_features = [(('count_op', False), 1.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)
    self.AssertFeaturesInByName(
      rule_features, not_expected_features, self.feat_instantiator, feat_in=False)

  @unittest.skipIf('count_op' not in FeatureInstantiator.feature_templates and \
                   'surface_overlap' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_CountPredicateFloorsTrue(self):
    rule = XTRule('q1',
                  tree_or_string('(ROOT (SBARQ (WHNP (WHADJP (WRB how) (JJ many)) (NNS floors)) ?x1|SQ))'),
                  tree_or_string('(ID !fb:architecture.building.floors ?x1|)'),
                  {(1,) : 'q2'}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    # expected_features = [(('count_op', True), 1.0), (('surface_match', '80<=ratio'), 1.0)]
    expected_features = [(('count_op', True), 1.0), ('surface_match', 1.0)]
    not_expected_features = [(('count_op', False), 1.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)
    self.AssertFeaturesInByName(
      rule_features, not_expected_features, self.feat_instantiator, feat_in=False)

  @unittest.skipIf('count_op' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_CountPredicateNumberTrue(self):
    rule = XTRule('q1',
                  tree_or_string('(ROOT (SBARQ (WHNP (WHADJP (WRB how) (JJ many)) ?x0|) ?x1|SQ (. ?)))'),
                  tree_or_string('(ID !fb:measurement_unit.dated_integer.number (ID ?x0| ?x1|))'),
                  {(0, 0): 'q2', (0, 1) : 'q2'}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    expected_features = [(('count_op', True), 1.0)]
    not_expected_features = [(('count_op', False), 1.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)
    self.AssertFeaturesInByName(
      rule_features, not_expected_features, self.feat_instantiator, feat_in=False)

  @unittest.skipIf('count_op' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_CountPredicateFalse(self):
    rule = XTRule('q1',
                  tree_or_string('(ROOT (SBARQ (WHNP (WHADJP (WRB how) (JJ many)) ?xx0|) ?x1|SQ (. ?)))'),
                  tree_or_string('(ID !fb:education.academic_post.institution ?x1|)'),
                  {(1,) : 'q2'}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    expected_features = [(('count_op', False), 1.0)]
    not_expected_features = [(('count_op', True), 1.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)
    self.AssertFeaturesInByName(
      rule_features, not_expected_features, self.feat_instantiator, feat_in=False)

  @unittest.skipIf('count_op' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_CountPredicateHowMuchFalse(self):
    rule = XTRule('q1',
                  tree_or_string('(ROOT (SBARQ (WHNP (WHADJP (WRB how) (JJ much)) ?xx0|) ?x1|SQ (. ?)))'),
                  tree_or_string('(ID !fb:architecture.building.floors ?x1|)'),
                  {(1,) : 'q2'}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    expected_features = [(('count_op', False), 1.0)]
    not_expected_features = [(('count_op', True), 1.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)
    self.AssertFeaturesInByName(
      rule_features, not_expected_features, self.feat_instantiator, feat_in=False)

  @unittest.skipIf('count_op' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_CountPredicateReverseFalse(self):
    rule = XTRule('q1',
                  tree_or_string('(ROOT (SBARQ (WHNP (WHADJP (WRB how) (JJ many)) ?xx0|) ?x1|SQ (. ?)))'),
                  tree_or_string('(ID fb:architecture.building.floors ?x1|)'),
                  {(1,) : 'q2'}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    expected_features = [(('count_op', False), 1.0)]
    not_expected_features = [(('count_op', True), 1.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator, feat_in=True)
    self.AssertFeaturesInByName(
      rule_features, not_expected_features, self.feat_instantiator, feat_in=False)

  @unittest.skipIf('type_match' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_TypeMatchWhoPerson(self):
    src_tree = tree_or_string('(ROOT (XX who) (YY was))')
    rule = XTRule('q1',
                  tree_or_string('(ROOT XX)'),
                  tree_or_string('(ID !fb:education.academic_post.person)'),
                  {}, 1.0)
    self.feat_instantiator.SetContext({'src_tree' : str(src_tree)})
    src_treep, trg_treep = rule.GetTreePatterns()
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    expected_features = [('who_person', 1.0)]
    not_expected_features = [('where_location', 1.0), ('when_date', 1.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)
    self.AssertFeaturesInByName(
      rule_features, not_expected_features, self.feat_instantiator, feat_in=False)

  @unittest.skipIf('type_match' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_TypeMatchWhoNoPerson(self):
    src_tree = tree_or_string('(ROOT (XX who) (YY was))')
    rule = XTRule('q1',
                  tree_or_string('(ROOT XX)'),
                  tree_or_string('(ID !fb:engineering.engine.energy_source)'),
                  {}, 1.0)
    self.feat_instantiator.SetContext({'src_tree' : str(src_tree)})
    src_treep, trg_treep = rule.GetTreePatterns()
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    expected_features = []
    not_expected_features = [('where_location', 1.0), ('when_date', 1.0), ('who_person', 1.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)
    self.AssertFeaturesInByName(
      rule_features, not_expected_features, self.feat_instantiator, feat_in=False)

  @unittest.skipIf('surf_to_uri_info' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_Surface2URIInfo(self):
    rule = XTRule('q1',
                  tree_or_string('(NP (DT the) (NNP nyse))'),
                  tree_or_string('fb:en.new_york_stock_exchange_inc'),
                  {}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    # expected_features = [(('surf_to_uri_info', '80<=ratio'), 1.0)]
    expected_features = [('surf_to_uri_info', 1.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)

  @unittest.skipIf('surf_to_uri_info' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_Surface2URIInfoLow(self):
    rule = XTRule('q1',
                  tree_or_string('(NP (DT the) (NNP yyyy))'),
                  tree_or_string('fb:en.new_york_stock_exchange_inc'),
                  {}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    # expected_features = [(('surf_to_uri_info', '0<=ratio<20'), 1.0)]
    expected_features = [('surf_to_uri_info', 0.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)

  @unittest.skipIf('src_context_to_uri' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_SrcContextToUriWhenDate(self):
    src_tree = tree_or_string(
      '(ROOT (SBARQ (WHADVP (WRB when)) (SQ (VBD was) (NP (NNP wells) (NNP fargo)) (VP (VBN founded))) (. ?)))')
    self.feat_instantiator.SetContext({'src_tree' : str(src_tree)})
    rule = XTRule('q1',
                  tree_or_string('(VP (VBN founded))'),
                  tree_or_string('!fb:organization.organization.date_founded'),
                  {}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    expected_features = [(('src_context_to_uri', ('when',)), 1.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)

  @unittest.skipIf('src_context_to_uri' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_SrcContextToUriWhereDate(self):
    src_tree = tree_or_string(
      '(ROOT (SBARQ (WHADVP (WRB where)) (SQ (VBD was) (NP (NNP wells) (NNP fargo)) (VP (VBN founded))) (. ?)))')
    self.feat_instantiator.SetContext({'src_tree' : str(src_tree)})
    rule = XTRule('q1',
                  tree_or_string('(VP (VBN founded))'),
                  tree_or_string('!fb:organization.organization.date_founded'),
                  {}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    not_expected_features = [(('src_context_to_uri', ('when',)), 1.0)]
    self.AssertFeaturesInByName(
      rule_features, not_expected_features, self.feat_instantiator, feat_in=False)

  @unittest.skipIf('src_context_to_uri' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_SrcContextToUriWhereLocation(self):
    src_tree = tree_or_string(
      '(ROOT (SBARQ (WHADVP (WRB where)) (SQ (VBD was) (NP (NNP william) (NNP shakespeare)) (VP (VBN born))) (. ?)))')
    self.feat_instantiator.SetContext({'src_tree' : str(src_tree)})
    rule = XTRule('q1',
                  tree_or_string('(VP (VBN born))'),
                  tree_or_string('!fb:people.person.place_of_birth'),
                  {}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    expected_features = [(('src_context_to_uri', ('where',)), 1.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)

  @unittest.skipIf('src_context_to_uri' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_SrcContextToUriWhereDate(self):
    src_tree = tree_or_string(
      '(ROOT (SBARQ (WHADVP (WRB where)) (SQ (VBD was) (NP (NNP william) (NNP shakespeare)) (VP (VBN born))) (. ?)))')
    self.feat_instantiator.SetContext({'src_tree' : str(src_tree)})
    rule = XTRule('q1',
                  tree_or_string('(VP (VBN born))'),
                  tree_or_string('!fb:people.person.date_of_birth'),
                  {}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    not_expected_features = [(('src_context_to_uri', ('where',)), 1.0)]
    self.AssertFeaturesInByName(
      rule_features, not_expected_features, self.feat_instantiator, feat_in=False)

  @unittest.skipIf('undeleted_how_many' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_UndeletedHowMany(self):
    rule = XTRule('q1',
                  tree_or_string('(ROOT (SBARQ (WHNP (WHADJP (WRB how) (JJ many)) ?x0|NNS) ?x1|SQ (. ?)))'),
                  tree_or_string('(ID COUNT (ID ?x0| ?x1|))'),
                  {(0, 0) : 'q2', (0, 1) : 'q2'}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    expected_features = [(('undeleted_how_many', True), 1.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)

  @unittest.skipIf('fb_typed_size' not in FeatureInstantiator.feature_templates,
    "feature not activated by default")
  def test_FbTypedSizeNoPredicate(self):
    rule = XTRule('q1',
                  tree_or_string('(VBD wrote)'),
                  tree_or_string('1234'),
                  {}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    rule_features = self.feat_instantiator.InstantiateFeatures(src_treep, trg_treep, rule=rule)
    expected_features = [('fb_typed_size', 0.0)]
    self.AssertFeaturesInByName(
      rule_features, expected_features, self.feat_instantiator)

if __name__ == '__main__':
  suite1  = unittest.TestLoader().loadTestsFromTestCase(LexicalFeaturesTestCase)
  suites  = unittest.TestSuite([suite1])
  unittest.TextTestRunner(verbosity=2).run(suites)

