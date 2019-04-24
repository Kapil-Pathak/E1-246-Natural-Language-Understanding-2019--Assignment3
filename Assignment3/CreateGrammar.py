import nltk
from nltk import Tree
from nltk.corpus import treebank
from nltk import Nonterminal, Production
from nltk import induce_pcfg

print("Induce PCFG grammar from treebank data:")

productions = []
Leaves=set()
for item in treebank.fileids():
    for tree in treebank.parsed_sents(item):
        tree.collapse_unary(collapsePOS = False)# Remove branches A-B-C into A-B+C
        tree.chomsky_normal_form(horzMarkov = 2)# Remove A->(B,C,D) into A->B,C+D->D
        productions += tree.productions()

from nltk  import Nonterminal
S = Nonterminal('S')
grammar = induce_pcfg(S, productions)
for i in range(len(grammar.productions())):
    pattern = re.sub(" -> ", "->", str(grammar.productions()[i]))
    print(pattern)
    print(i)
