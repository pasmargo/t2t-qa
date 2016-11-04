#!/bin/bash
# The original pipeline script obtains the following results:
# 1-best accuracy: 42 / 264 = 0.61
# oracle accuracy: 189 / 264 = 0.81

# Stages:
# 1. Data preparation.
# 2. Rule extraction.
# 3. Rule filtering.
# 4. Parameter estimation.
# 5. Decoding.
# 6. Evaluation.

initial_step=1
last_step=6

# Training max size: 641
ntrain=5
# Testing max size: 276
ntest=5

train_it=3
learning_rate=0.1
date_ini=`date`
fmt=json
model="perceptron_avg"
rem_qmark="--rem_qmark"

dec_timeout=800
cores=3
cores_dec=3

# Linkers based on leaf-alignments: ghkm_max, ghkm_cont, ghkm_giza
# Linkers based on dictionaries: gold_preds, gold
linker="gold_preds"
kgen_e=10
kgen_p=100
kdecoding=10000
exp_id=${2:-"emnlp2016"}

source /home/pasmargo/.virtualenvs/py2/bin/activate

echo "Experiment ID: "$exp_id" started on "$date_ini

stage=1
if [ "$initial_step" -le $stage ] && [ $stage -le "$last_step" ]; then
  echo $stage". Data preparation"

  rm -f data/qa.train.${exp_id}.tt*
  rm -f data/qa.train.${exp_id}.fttt.$fmt
  rm -f data/qa.train.${exp_id}.filtered.fttt.${fmt}
  rm -f data/qa.train.${exp_id}.fttt.${fmt}.descriptions
  rm -f data/qa.train.${exp_id}.wwfttt.tsv
  rm -f data/qa.test.${exp_id}.trg.json
  rm -f data/qa.test.${exp_id}.trg.err
  rm -f data/qa.test.${exp_id}.{tt,src}*
  rm -f data/evaluation.${exp_id}.trg.json
  rm -f data/evaluation.${exp_id}.results

  mkdir -p data

  python -m qald.prepare_data \
    --input qald/free917.train.enriched.examples.canonicalized.json \
    $rem_qmark \
    --random \
    | head -n $((ntrain * 2)) \
    > data/qa.train.${exp_id}.tt
  python -m qald.prepare_data \
    --input qald/free917.test.enriched.examples.canonicalized.json  \
    $rem_qmark \
    | head -n $((ntest * 2)) > data/qa.test.${exp_id}.tt
  awk '{if(NR % 2 == 1) {print $0}}' data/qa.test.${exp_id}.tt > data/qa.test.${exp_id}.src
fi

### Setting up cost functions ###
other_costs=" \
  @@LSWildrecog 1.0 \
  @@LSSize 0.05 \
  @@LSCount 5.0"
if [ "$linker" == "gold" ]; then
  rule_backoffs=" \
    @@LSDictent qald/qa.dict.txt.entities 1.0 \
    @@LSDictbent qald/qa.dict.txt.entities 1.0 \
    @@LSDictpred qald/qa.dict.txt.predicates 1.0"
elif [ "$linker" == "gold_preds" ]; then
  rule_backoffs=" \
    @@LSDictent qald/qa.dict.txt.entities_yates 1.0 \
    @@LSDictbent qald/qa.dict.txt.entities_yates 1.0 \
    @@LSDictpred qald/qa.dict.txt.predicates 1.0"
elif [ "$linker" == "ghkm_max" ]; then
  rule_backoffs=" \
    @@LSDictent qald/qa.dict.txt.entities_yates 1.0 \
    @@LSDictbent qald/qa.dict.txt.entities_yates 1.0 \
    @@LSDictpred qald/qa.dict.txt.predicates 1.0"
  python -m qald.align_max \
    --input data/qa.train.${exp_id}.tt \
    --entities qald/qa.dict.txt.entities_yates \
    --predicates qald/qa.dict.txt.predicates \
    --mode max \
    > data/qa.train.alignments.${exp_id}.txt
  other_costs=" \
    @@LSAlign data/qa.train.alignments.${exp_id}.txt 1.0 \
    @@NoStrings 1000 source \
    @@LSSize 10.0"
elif [ "$linker" == "ghkm_cont" ]; then
  rule_backoffs=" \
    @@LSDictent qald/qa.dict.txt.entities_yates 1.0 \
    @@LSDictbent qald/qa.dict.txt.entities_yates 1.0 \
    @@LSDictpred qald/qa.dict.txt.predicates 1.0"
  python -m qald.align_max \
    --input data/qa.train.${exp_id}.tt \
    --entities qald/qa.dict.txt.entities_yates \
    --predicates qald/qa.dict.txt.predicates \
    --mode max_contiguous \
    > data/qa.train.alignments.${exp_id}.txt
  other_costs=" \
    @@LSAlign data/qa.train.alignments.${exp_id}.txt 1.0 \
    @@NoStrings 1000 source \
    @@LSSize 10.0"
elif [ "$linker" == "ghkm_giza" ]; then
  rule_backoffs=" \
    @@LSDictent qald/qa.dict.txt.entities_yates 1.0 \
    @@LSDictbent qald/qa.dict.txt.entities_yates 1.0 \
    @@LSDictpred qald/qa.dict.txt.predicates 1.0"
  awk '{if(NR % 2 == 1) {print $0}}' data/qa.train.${exp_id}.tt > data/qa.train.${exp_id}.src
  awk '{if(NR % 2 == 0) {print $0}}' data/qa.train.${exp_id}.tt > data/qa.train.${exp_id}.trg
  paste -d'\n' data/qa.train.${exp_id}.{src,trg} data/qa.train.giza_alignments.txt \
    > data/qa.train.alignments.${exp_id}.txt
  other_costs=" \
    @@LSAlign data/qa.train.alignments.${exp_id}.txt 1.0 \
    @@NoStrings 1000 source \
    @@LSSize 10.0"
fi

let stage++
if [ "$initial_step" -le $stage ] && [ $stage -le "$last_step" ]; then
  echo -n $stage". Extracting rules"
  date_start=`date +%s`
  python -OO -m alignment.align \
    data/qa.train.${exp_id}.tt \
    @@LSNSQA 2.0 1.0 5.0 5.0 \
    $rule_backoffs \
    $other_costs \
    @@LSCSSE \
    @@kMaxSourceDepth 25 \
    @@kMaxSourceBranches 15 \
    @@phrase_length 2 \
    @@kBeamSize 10 \
    @@kDefaultRunTime 600 \
    @@cores $cores \
    @@fmtPrint $fmt \
    @@n_best 100 \
    @@deletions True \
    > data/qa.train.${exp_id}.fttt.$fmt
  date_end=`date +%s`
  echo "  ("$((date_end - date_start))" secs.)"
fi

let stage++
if [ "$initial_step" -le $stage ] && [ $stage -le "$last_step" ]; then
  echo $stage". Rule filtering."
  python -m utils.filter_rules \
    @o data/qa.train.${exp_id}.filtered.fttt.${fmt} \
    @@from $fmt \
    @@to $fmt \
    @@filter entity \
    @@filter predicate \
    @@filter bridge_entity \
    data/qa.train.${exp_id}.fttt.${fmt}
fi

let stage++
if [ "$initial_step" -le $stage ] && [ $stage -le "$last_step" ]; then
  echo -n $stage". Estimating weights using "$model" model "
  date_start=`date +%s`
  python -OO -m training.train_tree_to_tree_model \
    data/qa.train.${exp_id}.tt \
    data/qa.train.${exp_id}.filtered.fttt.${fmt} \
    @@model $model \
    @@insert_cvts \
    $rule_backoffs \
    @@fmtPrint $fmt \
    @@kMaxIterations $train_it \
    @@learning_rate $learning_rate \
    @@feat_inst data/qa.train.${exp_id}.fttt.${fmt}.descriptions \
    @@feat_names extraction/feature_names.txt \
    @@cores $cores \
    @@task qa \
    @o data/qa.train.${exp_id}.wwfttt.tsv
  echo
  date_end=`date +%s`
  echo "  ("$((date_end - date_start))" secs.)"
fi

let stage++
if [ "$initial_step" -le $stage ] && [ $stage -le "$last_step" ]; then
  echo -n $stage". Decoding"
  date_start=`date +%s`
  python -OO -m decoder.decode_qa_p \
    data/qa.test.${exp_id}.src \
    data/qa.train.${exp_id}.filtered.fttt.${fmt} \
    @@model perceptron data/qa.train.${exp_id}.wwfttt.tsv \
    @@insert_cvts \
    @@feat_inst data/qa.train.${exp_id}.fttt.${fmt}.descriptions \
    @@feat_names extraction/feature_names.txt \
    $rule_backoffs \
    @@fmtPrint $fmt \
    @@nbest ${kdecoding} \
    @@cores $cores_dec \
    @@timeout $dec_timeout \
    > data/qa.test.${exp_id}.trg.json \
    2> data/qa.test.${exp_id}.trg.err
  date_end=`date +%s`
  echo "  ("$((date_end - date_start))" secs.)"
fi

let stage++
if [ "$initial_step" -le $stage ] && [ $stage -le "$last_step" ]; then
  echo -n $stage". Evaluate"
  date_start=`date +%s`
  python -m qald.compare_results \
    data/qa.test.${exp_id}.trg.json \
    qald/free917.test.examples.canonicalized.json \
    > data/evaluation.${exp_id}.trg.json \
    2> data/evaluation.${exp_id}.results
  python -m qald.collect_stats \
    data/evaluation.${exp_id}.trg.json \
    >> data/evaluation.${exp_id}.results
  date_end=`date +%s`
  echo "  ("$((date_end - date_start))" secs.)"

  echo "Report sent by e-mail:"
  echo "Experiment ID: "$exp_id >> data/evaluation.${exp_id}.results
  date_end=`date`
  echo $date_ini" (started)" >> data/evaluation.${exp_id}.results
  echo $date_end" (completed)" >> data/evaluation.${exp_id}.results
  tail -n 10 data/evaluation.${exp_id}.results
fi
echo "Experiment ID: "$exp_id
