from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

#import ipdb as pdb
import nltk
nltk.data.path = ["/nfs/jsalt/share/nltk_data"] + nltk.data.path
# Install a few python packages using pip
from w266_common import utils

import pip
import pkgutil
from pip._internal import main as pipmain
if not pkgutil.find_loader("tqdm"):
    pipmain(["install", "tqdm"])
if not pkgutil.find_loader("graphviz"):
    pipmain(["install", "graphviz"])


import nltk
from  w266_common import treeviz
# Monkey-patch NLTK with better Tree display that works on Cloud or other display-less server.
print("Overriding nltk.tree.Tree pretty-printing to use custom GraphViz.")
treeviz.monkey_patch(nltk.tree.Tree, node_style_fn=None, format='svg')

import os, sys, collections
import copy
from importlib import reload

import numpy as np
import nltk
from nltk.tree import Tree
from IPython.display import display, HTML
from tqdm import tqdm as ProgressBar

import logging
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

# Helpers for this assignment
from w266_common import utils, treeviz
import part2_helpers
import pcfg, pcfg_test
import cky, cky_test

# Use the sample of the Penn Treebank included with NLTK.
assert(nltk.download('treebank'))
corpus = nltk.corpus.treebank

use_full_ptb = True 
if use_full_ptb:
    part2_helpers.verify_ptb_install()
    corpus = nltk.corpus.ptb  # Throws errors, for some reason
    # This configures the corpus to use the WSJ section only.
    # The Brown section has some mis-bracketed trees that will cause the 
    # corpus reader to throw (many) errors.
    if not hasattr(corpus, '_parsed_sents'):
        print("Monkey-patching corpus reader...")
        corpus._parsed_sents = corpus.parsed_sents
        corpus.parsed_sents = lambda: corpus._parsed_sents(categories=['news'])

print("Converting to common JSON format...")
print("Starting timer.")

import time
t_0 = time.time()

def find_depth(tree, subtree):
    treepositions = tree.treepositions()
    for indices in treepositions:
        if tree[indices] is subtree:
            return len(indices)
    raise runtime_error('something is wrong with implementation of find_depth')

#function converting Tree object to dictionary compatible with common JSON format
def sent_to_dict(sentence):
    json_d = {}

    text = ""
    for word in sentence.flatten():
        text += word + " "
    json_d["text"] = text

    max_height = sentence.height()
    for i, leaf in enumerate(sentence.subtrees(lambda t: t.height() == 2)): #modify the leafs by adding their index in the sentence
        leaf[0] = (leaf[0], str(i))
    targets = []
    for index, subtree in enumerate(sentence.subtrees()):
        assoc_words = subtree.leaves()
        assoc_words = [(i, int(j)) for i, j in assoc_words]
        assoc_words.sort(key=lambda elem: elem[1])
#        if subtree.label() == "NP" and subtree.leaves()[0][0] != '61' and find_depth(sentence, subtree) == 3:
#            pdb.set_trace()
        targets.append({"span1":[int(assoc_words[0][1]), int(assoc_words[-1][1]) + 1], "label":subtree.label(), "height": subtree.height() - 1, \
                        "depth": find_depth(sentence, subtree)})
    json_d["targets"] = targets
    
    json_d["info"] = {"source": "PTB"}
    
    return json_d

import json
data = {"data": []}
num_sent = len(corpus.parsed_sents())
#may want to parallelize this for loop
for sentence in corpus.parsed_sents():
    data["data"].append(sent_to_dict(sentence))

with open('ptb.json', 'w') as outfile:
    for datum in data["data"]:
        json.dump(datum, outfile)
        outfile.write("\n")

print("done.")
print("Converting to JSON takes " + str(time.time() - t_0) + " seconds.")
