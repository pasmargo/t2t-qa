import json
import sys

# Usage: python -m qald.collect_stats data/evaluation.full.cv1.small.trg.json
# acc     cov     empt.   preds   predsp  ents    entsp   brid    bridp   onevar  twovar  threev  total
# 0.15    0.20    0.45    11.20   0.40    3.85    3.80    0.00    0.00    5.85    1.25    0.00    22.15

def PrintStats(finput_fname):
  data = json.load(open(finput_fname))
  accuracy = 0.0
  coverage = 0.0
  empty_grammars = 0.0
  total_rules = 0.0
  total_valid = 0.0
  
  predicate_rules = 0.0
  predicate_pending_rules = 0.0
  entity_rules = 0.0
  entity_pending_rules = 0.0
  bridge_rules = 0.0
  bridge_pending_rules = 0.0
  one_var_rules = 0.0
  two_var_rules = 0.0
  three_var_rules = 0.0
  
  for d in data:
    if d['gold_result'] == "(list)" or \
       d['result_at_position'] == "n/a":
      continue
    total_valid += 1
    if d['result_at_position'] == 0:
      accuracy += 1
    if d['result_at_position'] >= 0:
      coverage += 1
    if d['status'] == 'empty_grammar':
      empty_grammars += 1
    if d['grammar']:
      total_rules += len(d['grammar'])
      predicate_rules += sum(["- state: predicate" in r for r in d['grammar']])
      predicate_pending_rules += sum(["- state: predicate_pending" in r for r in d['grammar']])
      entity_rules += sum(["- state: entity" in r for r in d['grammar']])
      entity_pending_rules += sum(["- state: entity_pending" in r for r in d['grammar']])
      bridge_rules += sum(["- state: bridge_entity" in r for r in d['grammar']])
      bridge_pending_rules += sum(["- state: bridge_entity_pending" in r for r in d['grammar']])
      for rule in d['grammar']:
        num_vars = sum([f.startswith('- [') for f in rule])
        if num_vars == 1:
          one_var_rules += 1
        if num_vars == 2:
          two_var_rules += 1
        if num_vars == 3:
          three_var_rules += 1

  if total_valid != 0:
    accuracy                = accuracy                / total_valid
    coverage                = coverage                / total_valid
    empty_grammars          = empty_grammars          / total_valid
    predicate_rules         = predicate_rules         / total_valid
    predicate_pending_rules = predicate_pending_rules / total_valid
    entity_rules            = entity_rules            / total_valid
    entity_pending_rules    = entity_pending_rules    / total_valid
    bridge_rules            = bridge_rules            / total_valid
    bridge_pending_rules    = bridge_pending_rules    / total_valid
    one_var_rules           = one_var_rules           / total_valid
    two_var_rules           = two_var_rules           / total_valid
    three_var_rules         = three_var_rules         / total_valid
    total_rules             = total_rules             / total_valid
  else:
    accuracy                = 0.0
    coverage                = 0.0
    empty_grammars          = 0.0
    predicate_rules         = 0.0
    predicate_pending_rules = 0.0
    entity_rules            = 0.0
    entity_pending_rules    = 0.0
    bridge_rules            = 0.0
    bridge_pending_rules    = 0.0
    one_var_rules           = 0.0
    two_var_rules           = 0.0
    three_var_rules         = 0.0
    total_rules             = 0.0

  print('acc\tcov\tempt.\tpreds\tpredsp\tents\tentsp\tbrid\tbridp\tonevar\ttwovar\tthreev\ttotal')
  print('{0:.2f}\t{1:.2f}\t{2:.2f}\t{3:.2f}\t{4:.2f}\t{5:.2f}\t{6:.2f}\t{7:.2f}\t{8:.2f}\t{9:.2f}\t{10:.2f}\t{11:.2f}\t{12:.2f}'.format(
        accuracy                ,
        coverage                ,
        empty_grammars          ,
        predicate_rules         ,
        predicate_pending_rules ,
        entity_rules            ,
        entity_pending_rules    ,
        bridge_rules            ,
        bridge_pending_rules    ,
        one_var_rules           ,
        two_var_rules           ,
        three_var_rules         ,
        total_rules             ))
  return accuracy, coverage, empty_grammars, predicate_rules, predicate_pending_rules, entity_rules, entity_pending_rules, bridge_rules, bridge_pending_rules, one_var_rules, two_var_rules, three_var_rules, total_rules

accuracy_acc = 0.0
coverage_acc = 0.0
empty_grammars_acc = 0.0
predicate_rules_acc = 0.0
predicate_pending_rules_acc = 0.0
entity_rules_acc = 0.0
entity_pending_rules_acc = 0.0
bridge_rules_acc = 0.0
bridge_pending_rules_acc = 0.0
one_var_rules_acc = 0.0
two_var_rules_acc = 0.0
three_var_rules_acc = 0.0
total_rules_acc = 0.0

finput_fnames = sys.argv[1:]
for finput_fname in finput_fnames:
  accuracy, coverage, empty_grammars, predicate_rules, predicate_pending_rules, entity_rules, entity_pending_rules, bridge_rules, bridge_pending_rules, one_var_rules, two_var_rules, three_var_rules, total_rules = PrintStats(finput_fname)

  accuracy_acc                += accuracy               
  coverage_acc                += coverage               
  empty_grammars_acc          += empty_grammars         
  predicate_rules_acc         += predicate_rules        
  predicate_pending_rules_acc += predicate_pending_rules
  entity_rules_acc            += entity_rules           
  entity_pending_rules_acc    += entity_pending_rules   
  bridge_rules_acc            += bridge_rules           
  bridge_pending_rules_acc    += bridge_pending_rules   
  one_var_rules_acc           += one_var_rules          
  two_var_rules_acc           += two_var_rules          
  three_var_rules_acc         += three_var_rules        
  total_rules_acc             += total_rules            

print('-' * 40)
print('acc\tcov\tempt.\tpreds\tpredsp\tents\tentsp\tbrid\tbridp\tonevar\ttwovar\tthreev\ttotal')
print('{0:.2f}\t{1:.2f}\t{2:.2f}\t{3:.2f}\t{4:.2f}\t{5:.2f}\t{6:.2f}\t{7:.2f}\t{8:.2f}\t{9:.2f}\t{10:.2f}\t{11:.2f}\t{12:.2f}'.format(
        accuracy_acc / len(finput_fnames),
        coverage_acc / len(finput_fnames),
        empty_grammars_acc / len(finput_fnames),
        predicate_rules_acc / len(finput_fnames),
        predicate_pending_rules_acc / len(finput_fnames),
        entity_rules_acc / len(finput_fnames),
        entity_pending_rules_acc / len(finput_fnames),
        bridge_rules_acc / len(finput_fnames),
        bridge_pending_rules_acc / len(finput_fnames),
        one_var_rules_acc / len(finput_fnames),
        two_var_rules_acc / len(finput_fnames),
        three_var_rules_acc / len(finput_fnames),
        total_rules_acc / len(finput_fnames)))
