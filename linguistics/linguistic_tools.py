#!/usr/bin/python

import codecs
from collections import defaultdict
import sys, os

import nltk
from nltk.corpus import wordnet as wn

from libs.functools27 import lru_cache

# Obtain lemmas of synonyms.
def ObtainSynonyms(word):
  return set([lemma for synonym in wn.synsets(word) \
                     for lemma in synonym.lemma_names()])

# Obtain lemmas of hypernyms.
def ObtainHypernyms(word):
  hyper = lambda s: s.hypernyms()
  return set([lemma for synonym in wn.synsets(word) \
                      for hypernym in synonym.closure(hyper) \
                        for lemma in hypernym.lemma_names()])

# def ObtainHypernyms(word):
#   return set([lemma for synonym in wn.synsets(word) \
#                       for hypernym in synonym.hypernyms() \
#                         for lemma in hypernym.lemma_names()])

# Obtain lemmas of hyponyms.
def ObtainHyponyms(word):
  return set([lemma for synonym in wn.synsets(word) \
                      for hyponym in synonym.hyponyms() \
                        for lemma in hyponym.lemma_names()])

# Obtain lemmas of holonyms.
def ObtainHolonyms(word):
  return set([lemma for synonym in wn.synsets(word) \
                      for holonym in synonym.member_holonyms() + \
                                     synonym.substance_holonyms() + \
                                     synonym.part_holonyms() \
                        for lemma in holonym.lemma_names()])

# Obtain lemmas of meronyms.
def ObtainMeronyms(word):
  return set([lemma for synonym in wn.synsets(word) \
                      for meronym in synonym.member_meronyms() + \
                                     synonym.substance_meronyms() + \
                                     synonym.part_meronyms() \
                        for lemma in meronym.lemma_names()])

# Obtain lemmas of antonyms.
def ObtainAntonyms(word):
  return set([antonym.name() for synonym in wn.synsets(word) \
                             for lemma in synonym.lemmas() \
                               for antonym in lemma.antonyms()])

# Obtain lemmas of entailments.
def ObtainEntailments(word):
  return set([lemma for synonym in wn.synsets(word) \
                      for entailment in synonym.entailments()\
                        for lemma in entailment.lemma_names()])

# Obtain derivationally related words.
def ObtainDerivations(word):
  return set([drf.name() for synonym in wn.synsets(word) \
                         for lemma in synonym.lemmas() \
                           for drf in lemma.derivationally_related_forms()])

# Returns the linguistic relationships of a word. E.g.
# ObtainLinguisticRelationships('friend') should return:
# [('antonym', 'foe'), ('synonym', 'mate'), ('hypernym', 'person'), etc.]
def ObtainLinguisticRelationships(word):
  word = word.strip('"')
  linguistic_relationships = []
  linguistic_relationships.append(('copy', word))
  base_word = wn.morphy(word)
  if base_word == None:
    base_word = word.lower()
  if word != base_word:
    linguistic_relationships.append(('inflection', base_word))
  linguistic_relationships.extend(\
    [('synonym', lemma) for lemma in ObtainSynonyms(word)])
  linguistic_relationships.extend(\
    [('hypernym', lemma) for lemma in ObtainHypernyms(word)])
  linguistic_relationships.extend(\
    [('hyponym', lemma) for lemma in ObtainHyponyms(word)])
  linguistic_relationships.extend(\
    [('holonym', lemma) for lemma in ObtainHolonyms(word)])
  linguistic_relationships.extend(\
    [('meronym', lemma) for lemma in ObtainMeronyms(word)])
  linguistic_relationships.extend(\
    [('antonym', lemma) for lemma in ObtainAntonyms(word)])
  linguistic_relationships.extend(\
    [('entailed', lemma) for lemma in ObtainEntailments(word)])
  linguistic_relationships.extend(\
    [('derivation', lemma) for lemma in ObtainDerivations(word)])
  return linguistic_relationships

# Check if word1 is synonym of word2, but checking whether the intersection
# between the synset of word1 and the synset of word2.
# If word1 = 'car' and word2 = 'automobile', this function should return True.
def IsSynonym(word1, word2):
  synsets_word1 = wn.synsets(word1)
  synsets_word2 = wn.synsets(word2)
  common_synset = set(synsets_word1).intersection(set(synsets_word2))
  if len(common_synset) != 0:
    return True
  return False

# Check whether word1 is a hypernym of word2, by computing the synset of word2,
# and for every possible meaning of word2, compute whether the hypernym of
# such meaning is in the list of synonyms (synset) of word1. E.g.
# IsHypernym('European', 'Swede') returns True.
def IsHypernym(word1, word2):
  synsets_word1 = wn.synsets(word1)
  synsets_word2 = wn.synsets(word2)
  for synonym in synsets_word2:
    for hypernym in synonym.hypernyms():
      if hypernym in synsets_word1:
        return True
  return False

# Check whether word1 is a hyponym of word2, by computing whether
# word2 is a hypernym of word1. E.g.
# IsHyponym('Swede', 'European') returns True.
def IsHyponym(word1, word2):
  return IsHypernym(word2, word1)

# Check whether word1 is a holonym of word2, by computing the synset of word2,
# for every meaning of word2 compute its holonym, and check if such holonym
# is in the list of meanings (synset) of word1. E.g.
# IsHolonym('door', 'lock') returns True.
def IsHolonym(word1, word2):
  synsets_word1 = wn.synsets(word1)
  synsets_word2 = wn.synsets(word2)
  for synonym in synsets_word2:
    holonyms = synonym.member_holonyms() + \
               synonym.substance_holonyms() + \
               synonym.part_holonyms()
    for holonym in holonyms:
      if holonym in synsets_word1:
        return True
  return False

# Check whether word1 is a meronym of word2, by computing whether
# word2 is a holonym of word1. E.g.
# IsMeronym('lock', 'door') returns True.
def IsMeronym(word1, word2):
  return IsHolonym(word2, word1)

# Checks whether word1 is an antonym of word2.
# Note that it may also consider antonyms words with different POS tags. E.g.
# IsAntonym('fast', 'slowly') returns True <- fast is adjective, slowly is adverb.
# IsAntonym('fast', 'slow') returns True.
# IsAntonym('good', 'bad') returns True.
# IsAntonym('good', 'poor') returns False.
def IsAntonym(word1, word2):
  synsets_word1 = wn.synsets(word1)
  synsets_word2 = wn.synsets(word2)
  antonyms = []
  for synonym in synsets_word1:
    for lemma in synonym.lemmas():
      antonyms.extend(lemma.antonyms()) # antonyms() method only works on lemmas.
  antonym_names = [antonym.name() for antonym in antonyms]
  for antonym_name in antonym_names:
    synsets_antonym = wn.synsets(antonym_name)
    common_meanings = set(synsets_word2).intersection(set(synsets_antonym))
    if len(common_meanings) > 0:
      return True
  return False

# Checks whether word1 is entailed by word2. E.g.
# IsEntailed('chew', 'eat') returns True. Only works on verbs.
def IsEntailed(word1, word2):
  synsets_word1 = wn.synsets(word1)
  synsets_word2 = wn.synsets(word2)
  for synonym in synsets_word2:
    entailments = synonym.entailments()
    if len(set(entailments).intersection(set(synsets_word1))):
      return True
  return False

# Snippet found in Stackoverflow.
def nounify(verb_word):
  """ Transform a verb to the closest noun: die -> death """
  verb_synsets = wn.synsets(verb_word, pos="v")
  # Word not found
  if not verb_synsets:
      return []
  # Get all verb lemmas of the word
  verb_lemmas = [l for s in verb_synsets \
                     for l in s.lemmas() if s.name().split('.')[1] == 'v']
  # Get related forms
  derivationally_related_forms = \
    [(l, l.derivationally_related_forms()) for l in verb_lemmas]
  # filter only the nouns
  related_noun_lemmas = [l for drf in derivationally_related_forms \
                         for l in drf[1] if l.synset.name().split('.')[1] == 'n']
  # Extract the words from the lemmas
  words = [l.name() for l in related_noun_lemmas]
  len_words = len(words)
  # Build the result in the form of a list containing tuples (word, probability)
  result = [(w, float(words.count(w))/len_words) for w in set(words)]
  result.sort(key=lambda w: -w[1])
  # return all the possibilities sorted by probability
  return result

# Check if word1 and word2 are inflectionally or derivationally related words,
# that is, words that had variations in their morphemes or that changed
# their POS category (e.g. converted from verb to noun).
def IsDerivation(word1, word2):
  synsets_word1 = wn.synsets(word1)
  # Word not found
  if not synsets_word1:
      return False
  # Get all lemmas of word1
  lemmas = [l for s in synsets_word1 for l in s.lemmas()]
  # Get related forms
  derivationally_related_forms = \
    [drf for l in lemmas for drf in l.derivationally_related_forms()]
  # Get a list of tuples, of the form [(lemma_name, POS), ...]
  lemma_pos = \
    [(l.name(), l.synset().name().split('.')[1]) for l in derivationally_related_forms]
  # Returns True if word2 is in the list of lemma names
  # that are derivations of word1.
  return (word2 in [l[0] for l in lemma_pos])

# Find linguistic relationship between two words.
# Remaining relationships that I would like to implement:
# LinguisticRelationship('man', 'men') would return 'plural'.
# LinguisticRelationship('go', 'went') would return 'present'.
# BUG: LinguisticRelationship('man', 'men') returns
#   ['synonym', 'hypernym', 'hyponym'] because 'man' and 'men' have the same
#   lemma but wn.morphy cannot recognize it. We should detect this and prevent
#   those relationships from triggering. However,
#   LinguisticRelationship('woman', 'women') returns ['inflection'] as expected,
#   until we implement the 'plural' relationship.
@lru_cache(maxsize = None)
def LinguisticRelationship(word1, word2):
  (word1, word2) = (word1.strip('"'), word2.strip('"'))
  if word1 == word2:
    return ['copy']
  base_word1 = wn.morphy(word1)
  base_word2 = wn.morphy(word2)
  if base_word1 == None:
    base_word1 = word1.lower()
  if base_word2 == None:
    base_word2 = word2.lower()
  ling_relations = []
  if word1 != word2 and base_word1 == base_word2:
    return ['inflection']
  if IsSynonym(base_word1, base_word2):
    ling_relations.append('synonym')
  if IsHypernym(base_word1, base_word2):
    ling_relations.append('hyponym')
  if IsHyponym(base_word1, base_word2):
    ling_relations.append('hypernym')
  if IsHolonym(base_word1, base_word2):
    ling_relations.append('holonym')
  if IsMeronym(base_word1, base_word2):
    ling_relations.append('meronym')
  if IsAntonym(base_word1, base_word2):
    ling_relations.append('antonym')
  if IsEntailed(base_word1, base_word2):
    ling_relations.append('entailed')
  if IsDerivation(word1, word2):
    ling_relations.append('derivation')
  return ling_relations

