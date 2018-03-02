#!/usr/bin/env python

from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

# Load Google's pre-trained Word2Vec model
model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True) 
# has plural and upper/lower case, and even bigrams (e.g., taxpayer_dollars; vast_sums)

# flex word2vec's muscles
model.doesnt_match("man woman child kitchen".split())
model.doesnt_match("france england germany berlin".split())
model.doesnt_match("paris berlin london austria".split())
model.most_similar("amsterdam")

# Consider a two-person task with a signaler and a receiver (similar to the TV gameshow 'Password'):
# The signalers were told that they would be playing a word-guessing game in which 
# they would have to think of one-word signals that would help someone guess their items. 
# They were talked through an example: if the item was 'dog', then a good signal would be 
# 'puppy' since most people given 'puppy' would probably guess 'dog'.

# sender thinks bank, says money
# receiver think cash
model.most_similar("bank") # .69 robber, .67 robbery, robbers, security, agency ..
model.most_similar("money") # .55 dollars, .55 profit, .54 cash
model.most_similar("cash") # .69 capitalize, .54 money, sell, debt, tax


model['money']

model.similarity("hot","cold") # .20
model.similarity("hot","warm") # .14

