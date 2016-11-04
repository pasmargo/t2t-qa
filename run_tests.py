import unittest

# Test clases for the training procedure.
from training.ruleindex_test import T2TGetRelevantRulesTestCase
from training.transductionrule_test import (ApplyRuleTestCase, RuleComparisonTestCase,
  PrintYamlTestCase, CopyRuleTestCase, PrintTiburonTestCase, ParseTiburonTestCase,
  RuleToTreePatternsTestCase)
from training.transducer_test import (ProduceTestCase, GetNonterminalsTestCase,
                                      TransduceTestCase, TransduceTestCase)
from training.wrtg_test import (PruneTestCase, InsideOutsideTestCase,
  GenerateTreesTestCase, TargetProjectionFromDerivationTestCase,
  SourceProjectionFromDerivationMixTestCase,
  SourceProjectionFromDerivationStrictTestCase, ObtainBestDerivationTestCase)
from training.train_perceptron_test import TrainPerceptronTestCase

# Test clases for the rule extraction procedure.
from extraction.extractor_test import (ExtractRulesTestCase, ObtainTreePatternTestCase,
  GetDisjointPathsTestCase, ExtractRulesDepsTestCase, GetCommonParentsAtTestCase,
  TransformationTestCase)
from extraction.feat_instantiator_test import LexicalFeaturesTestCase
from utils.priority_queue_test import PriorityQueueTestCase

# Test classes for tree utilities.
from utils.tree_tools_test import (TreePatternTestCase, GetPathsSpanningLeavesTestCase,
  GetInnerNodesTestCase, GetChildrenPathsTestCase, TreeContainsTestCase)

# Test classes for generator utilities.
from utils.generators_test import (GeneratorsListTestCase, PeekIterableTestCase,
  OrderedProductTestCase)

# Test classes for feature costs.
from linguistics.similarity_costs_deps_test import LexicalSimilarityDepsTestCase
from linguistics.similarity_pre_test import IdentitySimilarityTestCase
from linguistics.similarity_semantics_test import (InnerNodesDifferenceTestCase,
  VariableDifferenceTestCase, TreeDifferenceComplexityTestCase, TreeSizeTestCase,
  VariableDifferenceIndividualTestCase, EntityDifferenceIndividualTestCase)
from linguistics.similarity_qa_test import (NoSimilarityQATestCase, CountOpTestCase,
  EntityLinkingTestCase, PredicateLinkingTestCase, BridgeLinkingTestCase)
# Test classes for semirings.
from semirings.semiring_prob_test import ProbSemiRingTestCase

if __name__ == '__main__':
  suite1  = unittest.TestLoader().loadTestsFromTestCase(T2TGetRelevantRulesTestCase)
  suite2  = unittest.TestLoader().loadTestsFromTestCase(TreeContainsTestCase)
  suite3  = unittest.TestLoader().loadTestsFromTestCase(ApplyRuleTestCase)
  suite4  = unittest.TestLoader().loadTestsFromTestCase(RuleComparisonTestCase)
  suite6  = unittest.TestLoader().loadTestsFromTestCase(ProduceTestCase)
  suite7  = unittest.TestLoader().loadTestsFromTestCase(GetNonterminalsTestCase)
  suite8  = unittest.TestLoader().loadTestsFromTestCase(PruneTestCase)
  suite9  = unittest.TestLoader().loadTestsFromTestCase(InsideOutsideTestCase)
  suite11 = unittest.TestLoader().loadTestsFromTestCase(PrintYamlTestCase)
  suite12 = unittest.TestLoader().loadTestsFromTestCase(
    TargetProjectionFromDerivationTestCase)
  suite13 = unittest.TestLoader().loadTestsFromTestCase(GenerateTreesTestCase)
  suite14 = unittest.TestLoader().loadTestsFromTestCase(TransduceTestCase)
  suite15 = unittest.TestLoader().loadTestsFromTestCase(
    SourceProjectionFromDerivationMixTestCase)
  suite16 = unittest.TestLoader().loadTestsFromTestCase(ExtractRulesTestCase)
  suite17 = unittest.TestLoader().loadTestsFromTestCase(ObtainTreePatternTestCase)
  suite18 = unittest.TestLoader().loadTestsFromTestCase(GetDisjointPathsTestCase)
  suite19 = unittest.TestLoader().loadTestsFromTestCase(TreePatternTestCase)
  suite20 = unittest.TestLoader().loadTestsFromTestCase(LexicalSimilarityDepsTestCase)
  suite22 = unittest.TestLoader().loadTestsFromTestCase(
    SourceProjectionFromDerivationStrictTestCase)
  suite23 = unittest.TestLoader().loadTestsFromTestCase(ObtainBestDerivationTestCase)
  suite24 = unittest.TestLoader().loadTestsFromTestCase(ProbSemiRingTestCase)
  suite26 = unittest.TestLoader().loadTestsFromTestCase(CopyRuleTestCase)
  suite27 = unittest.TestLoader().loadTestsFromTestCase(IdentitySimilarityTestCase)
  suite28 = unittest.TestLoader().loadTestsFromTestCase(GetCommonParentsAtTestCase)
  suite29 = unittest.TestLoader().loadTestsFromTestCase(TransformationTestCase)
  suite30 = unittest.TestLoader().loadTestsFromTestCase(PriorityQueueTestCase)
  suite31 = unittest.TestLoader().loadTestsFromTestCase(GetPathsSpanningLeavesTestCase)
  suite32 = unittest.TestLoader().loadTestsFromTestCase(InnerNodesDifferenceTestCase)
  suite33 = unittest.TestLoader().loadTestsFromTestCase(VariableDifferenceTestCase)
  suite34 = unittest.TestLoader().loadTestsFromTestCase(GetInnerNodesTestCase)
  suite36 = unittest.TestLoader().loadTestsFromTestCase(TreeSizeTestCase)
  suite37 = unittest.TestLoader().loadTestsFromTestCase(GetChildrenPathsTestCase)
  suite38 = unittest.TestLoader().loadTestsFromTestCase(NoSimilarityQATestCase)
  suite42 = unittest.TestLoader().loadTestsFromTestCase(VariableDifferenceIndividualTestCase)
  suite43 = unittest.TestLoader().loadTestsFromTestCase(EntityDifferenceIndividualTestCase)
  suite45 = unittest.TestLoader().loadTestsFromTestCase(GeneratorsListTestCase)
  suite46 = unittest.TestLoader().loadTestsFromTestCase(PeekIterableTestCase)
  suite47 = unittest.TestLoader().loadTestsFromTestCase(OrderedProductTestCase)
  suite48 = unittest.TestLoader().loadTestsFromTestCase(PrintTiburonTestCase)
  suite49 = unittest.TestLoader().loadTestsFromTestCase(ParseTiburonTestCase)
  suite50 = unittest.TestLoader().loadTestsFromTestCase(TrainPerceptronTestCase)
  suite51 = unittest.TestLoader().loadTestsFromTestCase(RuleToTreePatternsTestCase)
  suite52 = unittest.TestLoader().loadTestsFromTestCase(LexicalFeaturesTestCase)
  suite53 = unittest.TestLoader().loadTestsFromTestCase(CountOpTestCase)

  suites  = unittest.TestSuite([
    suite1, suite2, suite3, suite4, suite6, suite7, suite8, suite9,
    suite11, suite12, suite13, suite14, suite15, suite16, suite17, suite18, suite19,
    suite20,
    suite22, suite23, suite24, suite26, suite27, suite28,
    suite29, suite30, suite31, suite32, suite33, suite34,
    suite36, suite37, suite38,
    suite42, suite43, suite45, suite46, suite47, suite48, suite49,
    suite50, suite51, suite52, suite53])
  unittest.TextTestRunner(verbosity=2).run(suites)

