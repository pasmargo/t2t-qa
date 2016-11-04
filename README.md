# Tree to Tree Tranducers for Question-Answering

This is a set of tools to use tree-to-tree transducers for Question-Answering. It includes:

* Automatic rule extraction that induces a tree transducer grammar.

* Cost functions that measure the cost of transforming source tree patterns into target tree patterns, useful to guide the heuristics that extract transformation rules.

* Implementation of the latent-variable structured averaged perceptron to estimate rule scores.

* Decoding routines that, given an input tree and a transducer grammar, it produces the set of all possible transformed trees. It uses back-off functions when necessary.

## Installation.

In order to run this software, you need python2.7 and some libraries that you can install using pip:
```bash
$ pip install -I nltk==3.0.0
$ pip install nltk pyyaml lxml mock numpy scipy simplejson SPARQLWrapper fuzzywuzzy py4j
```
If the installation of numpy or scipy fails, remember to install python-devel:
```bash
$ sudo apt-get install python-devel
```

In order to install Wornet, you can do:
```bash
$ python
>>> import nltk
>>> nltk.download()
```
Then select the tab Corpora > wordnet > click Download.

Some functions are written in cython for speed-up. Please, compile them using:

```bash
python setup.py build_ext --inplace
```

To run the QA pipeline, you need to install virtuoso and load the Freebase triplets. [SEMPRE](https://github.com/percyliang/sempre) includes a very clear explanations on how to do that, which I summarize here:

```bash
git clone https://github.com/percyliang/sempre.git
cd sempre
git checkout 6782fdb66b817034d4da06c675d3c9f37953e42f

# Then, install the dependencies (this will take a long time):
./download-dependencies core emnlp2013 acl2014 fullfreebase_vdb fullfreebase_ttl
make

# Clone into sempre folder
git clone https://github.com/openlink/virtuoso-opensource
cd virtuoso-opensource
git checkout 042f142
./autogen.sh
./configure --prefix=$PWD/install
make
make install
cd ..
```

Finally, you can run the tests to see if functions/methods are behaving as expected:
```bash
python -m run_tests
```
If there are no errors or failures displayed, you can proceed. Otherwise, there might be some software that you need to install beforehand. Please, e-mail me (and show me the test errors/failures) if you have any difficulty.

## Running the full pipeline

You can run the full pipeline (i.e. data preprocessing, rule extraction, parameter estimation, decoding and evaluation) with the following call:

```bash
$ ./run_qa.sh
```

If you want to speed up the pipeline, you can set the variables `cores` and `cores_dec` to the number of the desired cores to use. It parallelizes across tree pairs (at training) and questions (at testing).

When the script finishes, you will see a result similar to:

```
transpub$ ./run_qa.sh
Experiment ID: emnlp2016 started on Sat Nov 5 00:35:00 JST 2016
1. Data preparation
2. Extracting rules  (185 secs.)
3. Rule filtering.
4. Estimating weights using perceptron_avg model
ooooooooooooooooooooooooooooooooooooooooooooooooooxoxooooooooooooooooooooooooooooooxooooooooooooooooooooooooooooooooooooooooxoooooooooxoxoooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooxooooooooooooooooooooooooooooooooxooooxoooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooxooooooooxoooooooxooooooooooooooooooooooooooooooooooxoooooooxoooooooooooooooooooooxoooxoooooooooooooooooxoooooooooooooooxooooooooxooooooooooooooooooooxoooooooooooooooooooooooooooooooooooooooooooxoooooxoooooooooooooooooooooooooooooooooooooooooooooooxoooooooooooxoooooooooooooooooo
OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
........-..~.~~~~~~.~~~~~~..~~~...~~~~.~~~.~.~.~-.~~~~~..~~..~.~.~~~....~~~~~~~~~~~~~.~~~~.~~.~.~~~~~....~~-..-.~~~~~~~~.~.~~.~.~.~.~~~~...~.-~~~~~.~~~.~.~..~~~~.~~~~....~~.~.-~~.~~~...-~..~~..~.~..~~.~~~.~~~~~~~~~~.~~~...~~~.~.~~~-.~~...~~~.~~.~~~~~~......~~~.~.~~~~~..~~~..~~..~..-~.~...~.~..~~~~..~~~.~..~~.~~~~~~~~~~~..~~..~~..~~...~~~~~~..~...~~.~.~..~.~.~~~~~~~.~~.~~.-~~..~~..~~-~~~~~.~~.~....~.~~~~-..~~~~.~-~.~.~.~~~~-..~.~..~~~~...~~~~.~.~-~~.~~~~~~~~~-~~~~~..~~~~-~~~~~~~~~~~~~~~.....-.~.~~.~.~~.~~.-.~.~....-...~~.~.~.~-...~~..~~~~~~~..~~~.~-....~.~~.~..~~~~~~....~....~~..~~~~~-~-~~.~.~~~.~~~..~..~.~.....~~~.~~.~..~~~~..~.~~... Accuracy = 0.581903276131, error = 491.664807184
...~....-.~~~~~~~~..~~.~~~.~~.~~~~~~~~~~~.~~.~~~-~~~~.~~~~~~.~.~.~~~~~~~~~~~~~~~~.~~~~.~.~~~~.~~~~~~~~.~~~~-.~-~~~~~~~~~~~~~~~~~~.~.~.~~.~.~~-~~~~~~~...~.~.~~~~.~~~~.~~~.~.~~.-~~.~~~~.~-~~.~~.~~~.~~~~~~~~~.~~.~~~~~~~~~~~~~.~~~~.~~~-.~~~~~~~~~~~.~~~.~.~.~~~~~.~.~.~~~~~.~~~~.~~~~~~~.-~.~.~~~~.~..~~~~.~~~~~~.~..~.~~~.~~~~~~~~~.~~~.~~~.~~~~~~~...~...~~.~.~..~~.~.~~~~~.~~~~~~~-~~~~~.~..~-~.~~~..~.~.~~~~~~~~~-..~~~~~~-~~..~~.~~~-.~......~~.~~.~~~.~~~.-~~~~~~~~~~~~-~.~~..~~~~~-~~.~~~.~~~~~~~~....~-....~~~.~~.~.~-~~~~~.~.-~~~~~~~~..~-~...~~...~~~~~~~~~~..-~~.~~.~~.~~.~~...~~.~.~.~..~~.~~.~~~-~-~~.~~~~..~.~~~~~~~~~~..~.~.~~~~.~.~~~~~~~~.~~~.~ Accuracy = 0.705148205928, error = 258.505014675
.~~~~~~.-~~~~~~~~~~.~.~~~~~~~~~~~~~~~.~~~~~~.~~~-~~~~~~~~~~~~~~~.~~~~~~~~~~~~~~~~.~~~~~~~~~~~.~~~~~.~~~~~..-~.-~~.~~~~~~.~~~~.~~~.~~~.~~~~~~.-.~~~~~~...~.~.~~~.~.~.~~~~..~.~~.-~~~~~~~.~-~~.~~.~~~~~~~.~~~.~.~~~~~~~~~~~~~.~~~~~~~~~~~-.~~.~~~~~~.~..~~~~~~.~~~~~.~~~.~~~~~..~~~.~~~~~~~~-~~~~~.~~.~.~~~..~~~~~~~~~~~~~~~~.~.~~~.~~~~~~~~~~~.~~.~~~~~~~~...~~~~.~...~.~~~~~~~~~~~~~~~-~~.~~~~~.~-~~~~~~~~.~~~.~~~~~~~-.~.~~~~~-~~..~~~~~~-~...~~~~~~.~~.~~~~.~~~-~~~~~~~~~~~~-~.~~~.~~~~~-.~~~~~~~~~~~~~~~~..~-~~.~~~~~~~.~~~-.~~~~~.~-~~.~~~~~~.~-~..~~~~~~~~~~~..~~~.~-~~~.~~~~.~~.~~...~..~.~.~.~~~.~~.~.~-~-~~~~~~~~.~~~~~~~~~~~.....~~~.~~.~~.~~~~.~~.~~~.~ Accuracy = 0.769110764431, error = 153.166642349

  (2149 secs.)
5. Decoding  (3036 secs.)
6. Evaluate  (110 secs.)
Report sent by e-mail:
1-best accuracy: 168.0 / 264 = 0.64
oracle accuracy: 206.0 / 264 = 0.78
acc     cov     empt.   preds   predsp  ents    entsp   brid    bridp   onevar  twovar  threev  total
0.64    0.78    0.01    76.46   20.27   4.28    16.71   4.62    6.77    73.07   74.70   26.42   696.25
----------------------------------------
acc     cov     empt.   preds   predsp  ents    entsp   brid    bridp   onevar  twovar  threev  total
0.64    0.78    0.01    76.46   20.27   4.28    16.71   4.62    6.77    73.07   74.70   26.42   696.25
Experiment ID: gold_preds.e10-p100.ntrain641.ntest276
Sat Nov 5 00:35:00 JST 2016 (started)
Sat Nov 5 02:06:37 JST 2016 (completed)
Experiment ID: emnlp2016
```

The main logic of the rule extraction (as described in the paper) is in `extraction/extractor_beam.py`. Please write me if you would like to know more or would like to re-use any part of the pipeline (specially the rule extraction) for your own purposes.
