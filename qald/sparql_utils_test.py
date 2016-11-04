import unittest

from qald.sparql_utils import IsDisambiguator, Query

from lm.lm_qald import TypeCheckLM, GetLMScoreOfDerivations
from training.transductionrule import XTRule
from training.transducer import xT
from utils.tree_tools import tree_or_string
from utils.production import RHS, Production, TargetProjectionFromDerivation

class QueryFromLDCSCTestCase(unittest.TestCase):
  def setUp(self):
    self.prefix = 'PREFIX fb: <http://rdf.freebase.com/ns/>'

  def test_PredEnt(self):
    ldcsc = tree_or_string('(ID pred ent)')
    sparql = self.prefix + '\nSELECT DISTINCT ?x0 as ?answer WHERE {\n' \
           + '\t?x0\tpred\tent .} LIMIT 10 #'
    sparql_out = Query.fromldcsc(ldcsc)
    self.assertEqual(sparql, str(sparql_out),
      msg='\n{0}\n!=\n{1}'.format(sparql, sparql_out))

  def test_PredRevEnt(self):
    ldcsc = tree_or_string('(ID !pred ent)')
    sparql = self.prefix + '\nSELECT DISTINCT ?x0 as ?answer WHERE {\n' \
           + '\tent\tpred\t?x0 .} LIMIT 10 #'
    sparql_out = Query.fromldcsc(ldcsc)
    self.assertEqual(sparql, str(sparql_out),
      msg='\n{0}\n!=\n{1}'.format(sparql, sparql_out))

  def test_PredPredEnt(self):
    ldcsc = tree_or_string('(ID pred1 (ID pred2 ent))')
    sparql = self.prefix + '\nSELECT DISTINCT ?x0 as ?answer WHERE {\n' \
           + '\t?x0\tpred1\t?x1 .\n' \
           + '\t?x1\tpred2\tent .} LIMIT 10 #'
    sparql_out = Query.fromldcsc(ldcsc)
    self.assertEqual(sparql, str(sparql_out),
      msg='\n{0}\n!=\n{1}'.format(sparql, sparql_out))

  def test_PredRevPredEnt(self):
    ldcsc = tree_or_string('(ID !pred1 (ID pred2 ent))')
    sparql = self.prefix + '\nSELECT DISTINCT ?x0 as ?answer WHERE {\n' \
           + '\t?x1\tpred1\t?x0 .\n' \
           + '\t?x1\tpred2\tent .} LIMIT 10 #'
    sparql_out = Query.fromldcsc(ldcsc)
    self.assertEqual(sparql, str(sparql_out),
      msg='\n{0}\n!=\n{1}'.format(sparql, sparql_out))

  def test_PredRevPredRevEnt(self):
    ldcsc = tree_or_string('(ID !pred1 (ID !pred2 ent))')
    sparql = self.prefix + '\nSELECT DISTINCT ?x0 as ?answer WHERE {\n' \
           + '\t?x1\tpred1\t?x0 .\n' \
           + '\tent\tpred2\t?x1 .} LIMIT 10 #'
    sparql_out = Query.fromldcsc(ldcsc)
    self.assertEqual(sparql, str(sparql_out),
      msg='\n{0}\n!=\n{1}'.format(sparql, sparql_out))

  def test_Conjunction(self):
    ldcsc = tree_or_string('(ID (ID pred1 ent1) (ID pred2 ent2))')
    sparql = self.prefix + '\nSELECT DISTINCT ?x0 as ?answer WHERE {\n' \
           + '\t?x0\tpred1\tent1 .\n' \
           + '\t?x0\tpred2\tent2 .} LIMIT 10 #'
    sparql_out = Query.fromldcsc(ldcsc)
    self.assertEqual(sparql, str(sparql_out),
      msg='\n{0}\n!=\n{1}'.format(sparql, sparql_out))

  def test_PredPredEntPredEnt(self):
    ldcsc = tree_or_string('(ID pred1 (ID pred2 ent2) (ID pred3 ent3))')
    sparql = self.prefix + '\nSELECT DISTINCT ?x0 as ?answer WHERE {\n' \
           + '\t?x0\tpred1\t?x1 .\n' \
           + '\t?x1\tpred2\tent2 .\n' \
           + '\t?x1\tpred3\tent3 .} LIMIT 10 #'
    sparql_out = Query.fromldcsc(ldcsc)
    self.assertEqual(sparql, str(sparql_out),
      msg='\n{0}\n!=\n{1}'.format(sparql, sparql_out))

  def test_PredPredEntPredEnt(self):
    ldcsc = tree_or_string('(ID pred1 (ID !pred2 ent2) (ID pred3 ent3))')
    sparql = self.prefix + '\nSELECT DISTINCT ?x0 as ?answer WHERE {\n' \
           + '\t?x0\tpred1\t?x1 .\n' \
           + '\tent2\tpred2\t?x1 .\n' \
           + '\t?x1\tpred3\tent3 .} LIMIT 10 #'
    sparql_out = Query.fromldcsc(ldcsc)
    self.assertEqual(sparql, str(sparql_out),
      msg='\n{0}\n!=\n{1}'.format(sparql, sparql_out))

  def test_PredPredEntPredEntPolicyAll(self):
    ldcsc = tree_or_string('(ID pred1 (ID !pred2 ent2) (ID pred3 ent3))')
    sparql = self.prefix + '\nSELECT DISTINCT ?x0 as ?answer, ?x0, ?x1 WHERE {\n' \
           + '\t?x0\tpred1\t?x1 .\n' \
           + '\tent2\tpred2\t?x1 .\n' \
           + '\t?x1\tpred3\tent3 .} LIMIT 10 #'
    sparql_out = Query.fromldcsc(ldcsc, ['?x'])
    self.assertEqual(sparql, str(sparql_out),
      msg='\n{0}\n!=\n{1}'.format(sparql, sparql_out))

  def test_PredVarEnt(self):
    ldcsc = tree_or_string('(ID ?p1 ent)')
    sparql = self.prefix + '\nSELECT DISTINCT ?x0 as ?answer, ?p1 WHERE {\n' \
           + '\t?x0\t?p1\tent .} LIMIT 10 #'
    sparql_out = Query.fromldcsc(ldcsc, ['?p'])
    self.assertEqual(sparql, str(sparql_out),
      msg='\n{0}\n!=\n{1}'.format(sparql, sparql_out))

  def test_PredRevVarEnt(self):
    ldcsc = tree_or_string('(ID !?p1 ent)')
    sparql = self.prefix + '\nSELECT DISTINCT ?x0 as ?answer, ?p1 WHERE {\n' \
           + '\tent\t?p1\t?x0 .} LIMIT 10 #'
    sparql_out = Query.fromldcsc(ldcsc, ['?p'])
    self.assertEqual(sparql, str(sparql_out),
      msg='\n{0}\n!=\n{1}'.format(sparql, sparql_out))

  def test_PredRevVarEntPolicyAll(self):
    ldcsc = tree_or_string('(ID !?p1 ent)')
    sparql = self.prefix + '\nSELECT DISTINCT ?x0 as ?answer, ?p1 WHERE {\n' \
           + '\tent\t?p1\t?x0 .} LIMIT 10 #'
    sparql_out = Query.fromldcsc(ldcsc, ['?p'])
    self.assertEqual(sparql, str(sparql_out),
      msg='\n{0}\n!=\n{1}'.format(sparql, sparql_out))

  def test_OpPredVarEnt(self):
    ldcsc = tree_or_string('(ID COUNT (ID ?p1 ent))')
    sparql = self.prefix + '\nSELECT DISTINCT COUNT(?x0) as ?answer WHERE {\n' \
           + '\t?x0\t?p1\tent .} LIMIT 10 #'
    sparql_out = Query.fromldcsc(ldcsc)
    self.assertEqual(sparql, str(sparql_out),
      msg='\n{0}\n!=\n{1}'.format(sparql, sparql_out))

  def test_VarPredVarEnt(self):
    ldcsc = tree_or_string('(ID ?p1 (ID ?p2 ent))')
    sparql = self.prefix + '\nSELECT DISTINCT ?x0 as ?answer, ?p1, ?p2 WHERE {\n' \
           + '\t?x0\t?p1\t?x1 .\n' \
           + '\t?x1\t?p2\tent .} LIMIT 10 #'
    sparql_out = Query.fromldcsc(ldcsc, ['?p'])
    self.assertEqual(sparql, str(sparql_out),
      msg='\n{0}\n!=\n{1}'.format(sparql, sparql_out))

  def test_ComplexDate(self):
    ldcsc = tree_or_string(
      '(ID !fb:award.ranking.rank ' \
        + '(ID fb:award.ranking.year ' \
        +    '(DATE 2000_-1_-1)) ' \
        + '(ID fb:award.ranking.list fb:en.fortune_500) ' \
        + '(ID fb:award.ranking.item fb:en.monsanto))')
    sparql = self.prefix + '\nSELECT DISTINCT ?x0 as ?answer WHERE {\n' \
           + '\t?x1\tfb:award.ranking.rank\t?x0 .\n' \
           + 'FILTER (xsd:dateTime(?d0) >= xsd:dateTime("2000"^^xsd:datetime)) .\n' \
           + 'FILTER (xsd:dateTime(?d0) < xsd:dateTime("2001"^^xsd:datetime)) .\n' \
           + '\t?x1\tfb:award.ranking.year\t?d0 .\n' \
           + '\t?x1\tfb:award.ranking.list\tfb:en.fortune_500 .\n' \
           + '\t?x1\tfb:award.ranking.item\tfb:en.monsanto .} LIMIT 10 #'
    sparql_out = Query.fromldcsc(ldcsc)
    self.assertEqual(sparql, str(sparql_out),
      msg='\n{0}\n!=\n{1}'.format(sparql, sparql_out))

  def test_ComplexNumber(self):
    ldcsc = tree_or_string(
      '(ID fb:government.us_president.presidency_number ' \
        + '(NUMBER 23.0 fb:en.unitless))')
    sparql = self.prefix + '\nSELECT DISTINCT ?x0 as ?answer WHERE {\n' \
           + '\t?x0\tfb:government.us_president.presidency_number\t23.0 .} LIMIT 10 #'
    sparql_out = Query.fromldcsc(ldcsc)
    self.assertEqual(sparql, str(sparql_out),
      msg='\n{0}\n!=\n{1}'.format(sparql, sparql_out))

  def test_NumberWrongFormat(self):
    ldcsc = tree_or_string('(ID pred1 (NUMBER (ID pred2 ent2)))')
    sparql = self.prefix + '\nSELECT DISTINCT ?x0 as ?answer WHERE {\n' \
           + '\t?x0\tpred1\t?n0 .} LIMIT 10 #'
    sparql_out = Query.fromldcsc(ldcsc)
    self.assertEqual(sparql, str(sparql_out),
      msg='\n{0}\n!=\n{1}'.format(sparql, sparql_out))

  def test_DateWrongFormat(self):
    ldcsc = tree_or_string('(DATE (ID pred1 ent1))')
    sparql_out = Query.fromldcsc(ldcsc)
    self.assertIsNone(sparql_out,
      msg='\n{0}\nis not None'.format(sparql_out))

# PREFIX fb: <http://rdf.freebase.com/ns/>
# SELECT DISTINCT ?x3 WHERE {
#   FILTER (xsd:dateTime(?x2) >= xsd:dateTime("2000"^^xsd:datetime)) .
#   FILTER (xsd:dateTime(?x2) < xsd:dateTime("2001"^^xsd:datetime)) .
#   ?x1 fb:award.ranking.year ?x2 .
#   FILTER (?x1 != "2000"^^xsd:datetime) .
#   ?x1 fb:award.ranking.list fb:en.fortune_500 .
#   FILTER (?x1 != fb:en.fortune_500) .
#   ?x1 fb:award.ranking.item fb:en.monsanto .
#   FILTER (?x1 != fb:en.monsanto) .
#   ?x1 fb:award.ranking.rank ?x3
# } LIMIT 10 #

# PREFIX fb: <http://rdf.freebase.com/ns/>
# SELECT DISTINCT ?x1 ?x1name WHERE {
#   ?x1 fb:government.us_president.presidency_number 23.0 .
#   OPTIONAL {
#     ?x1 fb:type.object.name ?x1name
#   }
# } LIMIT 10 #

  # def test_Real(self):
  #   ldcsc = tree_or_string(
  #     '(ID ?p0 (ID !fb:fictional_universe.fictional_character.character_created_by fb:en.professor_challenger))')
  #   sparql_out = Query.fromldcsc(ldcsc, ['?p'])
  #   print(sparql_out)
  #   for result in sorted(sparql_out.get_results(), key=lambda x: x[1]):
  #     print(result)

class IsDisambiguatorTestCase(unittest.TestCase):
  def test_RelPositive(self):
    rel = 'fb:education.academic_post.person'
    self.assertTrue(IsDisambiguator(rel))

  def test_RelNegative(self):
    rel = 'fb:automotive.trim_level.max_passengers'
    self.assertFalse(IsDisambiguator(rel))

  def test_RelNegative2(self):
    rel = 'fb:book.technical_report.institution'
    self.assertFalse(IsDisambiguator(rel))

  def test_EntNegative(self):
    rel = 'fb:en.marshall_hall'
    self.assertFalse(IsDisambiguator(rel))

if __name__ == '__main__':
  suite1  = unittest.TestLoader().loadTestsFromTestCase(IsDisambiguatorTestCase)
  suite2  = unittest.TestLoader().loadTestsFromTestCase(QueryFromLDCSCTestCase)
  suites  = unittest.TestSuite([suite1, suite2])
  unittest.TextTestRunner(verbosity=2).run(suites)

