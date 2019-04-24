import nltk
from nltk import Tree
from nltk.corpus import treebank
from nltk import Nonterminal, Production
from nltk import induce_pcfg
import re
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
f=open("G.cfg","w+")
for i in range(len(grammar.productions())):
    pattern =str(grammar.productions()[i])
    left=grammar.productions()[i].lhs()
    right=grammar.productions()[i].rhs()
    p1=float(pattern.split(' -> ')[1].split()[-1][1:-1])
    if len(list(right))==1:
        print(str(left)+" -> "+str(right[0])+" :: "+str(p1))
        f.write(str(left)+" -> "+str(right[0])+" :: "+str(p1)+"\n")
    else:
        print(str(left)+" -> "+str(right[0])+" "+str(right[1])+" :: "+str(p1))
        f.write(str(left)+" -> "+str(right[0])+" "+str(right[1])+" :: "+str(p1)+"\n")
