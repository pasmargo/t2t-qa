import codecs
import logging
import sys, os, argparse, traceback

from training.loadrules import loadrules, GetInitialState
from utils.filter_rules_utils import FilterOutRulesWithCVT, FilterOutRulesByStates

class ConvertRulesParser:

  def __init__(self, option):
    self.__input                = option.input
    self.__output               = option.output
    self.__from_fmt             = option.from_fmt
    self.__to_fmt               = option.to_fmt
    self.__filter_states        = option.filter_states
    self.__rem_cvt              = option.rem_cvt

  def run(self):
    num_min_occur = 0
    rules = loadrules(self.__input, self.__from_fmt, num_min_occur)
    initial_state = GetInitialState(self.__input, self.__from_fmt)

    foutput = codecs.open(self.__output, 'w', 'utf-8')

    # Write initial state.
    if self.__to_fmt == 'json':
      foutput.write(u'{"general_info": {"initial_state": "' + initial_state + '"}}\n\n')
    elif self.__to_fmt == 'yaml':
      foutput.write(u'# initial_state: ' + initial_state + '\n\n')
    elif self.__to_fmt == 'tiburon':
      foutput.write(initial_state + u'\n\n')

    # Filtering.
    remaining_rules = FilterOutRulesByStates(rules, self.__filter_states)
    if self.__rem_cvt:
      remaining_rules = FilterOutRulesWithCVT(remaining_rules)

    # Write rules.
    for rule in remaining_rules:
      if self.__to_fmt == 'json':
        foutput.write(rule.PrintJson() + '\n')
      elif self.__to_fmt == 'yaml':
        foutput.write(rule.PrintYaml() + '\n\n')
      elif self.__to_fmt == 'tiburon':
        rule.weight = None
        foutput.write(rule.PrintTiburon() + '\n')

    foutput.close()

def main(args = None):
  import textwrap
  usage = "usage: %prog [options]"
  parser = argparse.ArgumentParser(usage)
  parser = argparse.ArgumentParser(
    prefix_chars='@',
    formatter_class=argparse.RawDescriptionHelpFormatter, 
    description=textwrap.dedent('''\
    rules.{json,yaml,tib} must contain an initial state and a list of rules.
    '''))

  parser.add_argument('input', nargs='?', type=str, default=sys.stdin, metavar="INPUT", \
    help="Input rules file (i.e., rules.yaml).")
  parser.add_argument("@o", "@@output", dest="output", nargs='?', type=str,
    default="output.rules", metavar="output.rules", \
    help="Converted rules in json, yaml or tiburon format.")
  parser.add_argument("@@from",	dest="from_fmt", nargs='?', type=str, default="json", \
    help="Set the format of the original rules file (json, yaml or tiburon). Default is json.")
  parser.add_argument("@@to", dest="to_fmt", nargs='?', type=str, default="yaml", \
    help="Set the format of the converted rules file (json, yaml or tiburon). Default is yaml.")
  parser.add_argument("@@filter", action='append', dest="filter_states", default=[], \
    help="List states for which rules will be filtered out. Default is empty.")
  parser.add_argument("@@rem_cvt", action="store_true", default=False,
    help="Remove rules that contain CVTs on RHS.")
  args = parser.parse_args()

  logging.basicConfig(level=logging.WARNING)

  converter = ConvertRulesParser(args)
  converter.run()


if __name__ == '__main__':
  try:
    main()
  except Exception as e:
    traceback.print_exc(file=sys.stderr)
    sys.exit(255)
