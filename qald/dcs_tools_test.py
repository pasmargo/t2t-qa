import unittest

# from pudb import set_trace; set_trace()

from nltk import Tree

from qald.dcs_tools import dcs2constituent, constituent2dcs
from utils.tree_tools import tree_or_string

class DCS2ConstituentTestCase(unittest.TestCase):
  """
  Here we test whether we can revert the dcs2constituent conversion
  back into a lambda-DCS tree.
  """
  def test_Unary(self):
    dcs = Tree.fromstring('(!fb:tv.tv_series_episode.writer fb:en.straight_and_true)')
    constituent = dcs2constituent(dcs)[0]
    expected_constituent = \
      Tree.fromstring('(ID !fb:tv.tv_series_episode.writer fb:en.straight_and_true)')
    self.assertEqual(expected_constituent, constituent)

  def test_Binary(self):
    dcs = tree_or_string('!fb:tv.tv_series_episode.writer')
    constituent = dcs2constituent(dcs)[0]
    expected_constituent = \
      tree_or_string('!fb:tv.tv_series_episode.writer')
    self.assertEqual(expected_constituent, constituent)

  def test_Count(self):
    dcs = Tree.fromstring('(count (!fb:military.armed_force.units fb:en.u_army))')
    # from pudb import set_trace; set_trace()
    constituent = dcs2constituent(dcs)[0]
    expected_constituent = \
      Tree.fromstring('(ID COUNT (ID !fb:military.armed_force.units fb:en.u_army))')
    self.assertEqual(expected_constituent, constituent)

  def test_Number(self):
    dcs = Tree.fromstring('(fb:government.us_president.presidency_number (number 22.0 fb:en.unitless))')
    constituent = dcs2constituent(dcs)[0]
    expected_constituent = \
      Tree.fromstring('(ID fb:government.us_president.presidency_number (NUMBER 22.0 fb:en.unitless))')
    self.assertEqual(expected_constituent, constituent)

  def test_Date(self):
    dcs = Tree.fromstring('(fb:soccer.football_team_management_tenure.to (date 2004 -1 -1))')
    constituent = dcs2constituent(dcs)[0]
    expected_constituent = \
      Tree.fromstring('(ID fb:soccer.football_team_management_tenure.to (DATE 2004_-1_-1))')
    self.assertEqual(expected_constituent, constituent)

  def test_And(self):
    dcs = Tree.fromstring('(pred1 (and pred2 pred3))')
    constituent = dcs2constituent(dcs)[0]
    expected_constituent = \
      Tree.fromstring('(ID pred1 pred2 pred3)')
    self.assertEqual(expected_constituent, constituent)

  @unittest.expectedFailure
  def test_AndNoParentPredicate(self):
    dcs = Tree.fromstring('(and fb:en.doom (fb:cvg.computer_videogame.gameplay_modes fb:en.multiplayer_game))')
    constituent = dcs2constituent(dcs)[0]
    expected_constituent = \
      Tree.fromstring('(ID fb:en.doom (ID fb:cvg.computer_videogame.gameplay_modes fb:en.multiplayer_game))')
    self.assertEqual(expected_constituent, constituent)

  def test_LambdaApplication(self):
    dcs = Tree.fromstring('((lambda x (pred2 (var x))) pred3)')
    constituent = dcs2constituent(dcs)[0]
    expected_constituent = \
      Tree.fromstring('(ID pred2 pred3)')
    self.assertEqual(expected_constituent, constituent)

  def test_LambdaApplicationParentPredicate(self):
    dcs = Tree.fromstring('(pred1 ((lambda x (pred2 (var x))) pred3))')
    constituent = dcs2constituent(dcs)[0]
    expected_constituent = \
      Tree.fromstring('(ID pred1 (ID pred2 pred3))')
    self.assertEqual(expected_constituent, constituent)

  def test_LambdaNoArgument(self):
    dcs = Tree.fromstring('(lambda x (pred2 (var x)))')
    constituent = dcs2constituent(dcs)[0]
    expected_constituent = \
      tree_or_string('(ID pred2)')
    self.assertEqual(str(expected_constituent), str(constituent))

  def test_LambdaComplex(self):
    dcs = Tree.fromstring('((lambda x (pred1 (pred2 (var x)))) pred3)')
    constituent = dcs2constituent(dcs)[0]
    expected_constituent = \
      Tree.fromstring('(ID pred1 (ID pred2 pred3))')
    self.assertEqual(expected_constituent, constituent)

  @unittest.expectedFailure
  def test_LambdaComplexNoArgument(self):
    dcs = Tree.fromstring('(lambda x (pred1 (pred2 (var x))))')
    constituent = dcs2constituent(dcs)[0]
    expected_constituent = \
      Tree.fromstring('(ID pred1 pred2)')
    self.assertEqual(expected_constituent, constituent)

  def test_JoinAnd(self):
    dcs = Tree.fromstring(('(!fb:education.academic_post.institution'
                           ' (and (fb:education.academic_post.person fb:en.marshall_hall)'
                                ' (fb:education.academic_post.position_or_title fb:en.professor)))'))
    constituent = dcs2constituent(dcs)[0]
    expected_constituent = \
      Tree.fromstring(('(ID !fb:education.academic_post.institution'
                       ' (ID fb:education.academic_post.person fb:en.marshall_hall)'
                       ' (ID fb:education.academic_post.position_or_title fb:en.professor))'))
    self.assertEqual(expected_constituent, constituent)

class Constituent2DCSTestCase(unittest.TestCase):
  """
  Here we test whether we can revert the dcs2constituent conversion
  back into a lambda-DCS tree.
  """
  def test_Unary(self):
    dcs = Tree.fromstring('(!fb:tv.tv_series_episode.writer fb:en.straight_and_true)')
    constituent = dcs2constituent(dcs)[0]
    expected_dcs = constituent2dcs(constituent)[0]
    self.assertEqual(expected_dcs, dcs)

  def test_Count(self):
    dcs = Tree.fromstring('(count (!fb:military.armed_force.units fb:en.u_army))')
    constituent = dcs2constituent(dcs)[0]
    expected_dcs = constituent2dcs(constituent)[0]
    self.assertEqual(expected_dcs, dcs)

  def test_Date(self):
    dcs = Tree.fromstring('(fb:soccer.football_team_management_tenure.to (date 2004 -1 -1))')
    constituent = dcs2constituent(dcs)[0]
    expected_dcs = constituent2dcs(constituent)[0]
    self.assertEqual(expected_dcs, dcs)

  def test_Number(self):
    dcs = Tree.fromstring('(fb:government.us_president.presidency_number (number 22.0 fb:en.unitless))')
    constituent = dcs2constituent(dcs)[0]
    expected_dcs = constituent2dcs(constituent)[0]
    self.assertEqual(expected_dcs, dcs)

  @unittest.expectedFailure
  def test_And(self):
    dcs = Tree.fromstring('(and fb:en.doom (fb:cvg.computer_videogame.gameplay_modes fb:en.multiplayer_game))')
    constituent = dcs2constituent(dcs)[0]
    expected_dcs = constituent2dcs(constituent)[0]
    self.assertEqual(expected_dcs, dcs)

  def test_JoinAnd(self):
    constituent = Tree.fromstring(
      ('(ID !fb:education.academic_post.institution'
       ' (ID fb:education.academic_post.person fb:en.marshall_hall)'
       ' (ID fb:education.academic_post.position_or_title fb:en.professor))'))
    expected_dcs = Tree.fromstring(
      ('(!fb:education.academic_post.institution'
       ' (and (fb:education.academic_post.person fb:en.marshall_hall)'
            ' (fb:education.academic_post.position_or_title fb:en.professor)))'))
    dcs = constituent2dcs(constituent)[0]
    self.assertEqual(expected_dcs, dcs)

if __name__ == '__main__':
  suite1  = unittest.TestLoader().loadTestsFromTestCase(Constituent2DCSTestCase)
  suite2  = unittest.TestLoader().loadTestsFromTestCase(DCS2ConstituentTestCase)
  suites  = unittest.TestSuite([suite1, suite2])
  unittest.TextTestRunner(verbosity=2).run(suites)

