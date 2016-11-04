
## Tokens to exclude.
# Natural language.
prepositions = ["in", "on", "of", "for", "about", "at", "from", "to", "with", "as"]
be_variations = ["'m", "am", "'re", "are", "'s", "is", "was", "were", "be", "being"]
modals = ["will"]
quasi_modals = ["used_to"]
determiners = ["the", "a", "an"]
adverbs = ["also", "very", "currently", "originally", "really"]
qwords = ["what", "how", "where", "when", "who", "why"]
# URI segments.
uri_segments = ['fb', 'en', 'm']

exclude_words = prepositions + be_variations + modals + quasi_modals + \
  determiners + adverbs + uri_segments + qwords

## Tokens to expand.
token_expansion = {
  'about' : ['subject'],
  'when' : ['date'],
  'where' : ['area', 'country', 'location', 'place', 'site'],
  'born' : ['bear', 'birth'],
  'old' : ['age', 'years'],
  'cost' : ['money', 'price', 'fee'],
  'area' : ['country']
}

def filter_tokens(tokens):
  filtered_tokens = [t for t in tokens if t not in exclude_words]
  return filtered_tokens

