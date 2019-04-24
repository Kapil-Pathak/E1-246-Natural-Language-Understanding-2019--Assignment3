
# coding: utf-8

# In[1]:


import nltk
from nltk import Tree
from nltk.corpus import treebank
from nltk import Nonterminal, Production
from nltk import induce_pcfg


# In[2]:


print("Induce PCFG grammar from treebank data:")

productions = []
Leaves=set()
for item in treebank.fileids():
    for tree in treebank.parsed_sents(item):
        tree.collapse_unary(collapsePOS = False)# Remove branches A-B-C into A-B+C
        tree.chomsky_normal_form(horzMarkov = 2)# Remove A->(B,C,D) into A->B,C+D->D
        productions += tree.productions()
        for i in range(len(tree.leaves())):
            Leaves.add(tree.leaves()[i])


# In[3]:


Leaves=list(Leaves)


# In[4]:


from nltk  import Nonterminal
S = Nonterminal('S')
grammar = induce_pcfg(S, productions)


# In[9]:


from collections import defaultdict
print(grammar.productions()[0].lhs())
source=defaultdict()
prob=defaultdict()
rhs_1=set()
lhs_1=set()
total=set()
for i in range(len(grammar.productions())):
    q=len(grammar.productions()[i].rhs())
    print(grammar.productions()[i].lhs())
    A=str(grammar.productions()[i]).split(' -> ')[0]
    
    BC=str(grammar.productions()[i]).split(' -> ')[1].split()[:-1]
    source[A]=BC
    lhs_1.add(A)
    total.add(A)
    p1=float(str(grammar.productions()[i]).split(' -> ')[1].split()[-1][1:-1])
    if len(BC)==1:
        prob[str(grammar.productions()[i]).split(' -> ')[0]+" -> "+str(BC[0])]=p1
        rhs_1.add(BC[0])
        total.add(BC[0])
        print(BC[0][1:-1])
        break
        print(BC[0])
    elif len(BC)==2:
        prob[str(grammar.productions()[i]).split(' -> ')[0]+" -> "+str(BC[0])+" "+str(BC[1])]=p1
        rhs_1.add(BC[0])
        rhs_1.add(BC[1])
        print(BC[0])
        print(BC[1])
        total.add(BC[0])
        total.add(BC[1])
rhs_dict=defaultdict()
lhs_dict=defaultdict()
rhs_dict_r=defaultdict()
lhs_dict_l=defaultdict()
for i, item in enumerate(lhs_1):
    lhs_dict[item]=i
    lhs_dict_l[i]=item
total_dict=defaultdict()
reverse_dict=defaultdict()
for i, item in enumerate(total):
    total_dict[item]=i
    reverse_dict[i]=item


# In[ ]:


Unary={}
Binary={}
for i in range(len(grammar.productions())):
    if len(grammar.productions()[i].rhs())==1:
        A=str(grammar.productions()[i]).split(' -> ')[0]
        BC=str(grammar.productions()[i]).split(' -> ')[1].split()[:-1]
        if A in Unary.keys():
            Unary[A].append(BC[0])
        else:
            Unary[A]=[]
            Unary[A].append(BC[0])
    else:
        A=str(grammar.productions()[i]).split(' -> ')[0]
        BC=str(grammar.productions()[i]).split(' -> ')[1].split()[:-1]
        if A in Binary.keys():
            Binary[A].append(BC)
        else:
            Binary[A]=[]
            Binary[A].append(BC)


# In[ ]:


Binary


# In[ ]:


total_dict['VP|<VP-PP>']
total1=list(total)


# In[ ]:


import numpy as np
from tqdm import tqdm_notebook as tqdm
sent="'I' 'saw' 'John' 'with' 'my' 'eyes'"
sent=sent.split(" ")
print(sent)
nonterms=['CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','SYM','TO','UH','VBD','VB','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB']
nonterm=len(lhs_1)
score = [ [ [0.0 for y in range(nonterm ) ] for x in range( len(sent) +1 ) ] for z in range(len(sent) +1)]
back = [ [ [None for y in range( nonterm) ] for x in range( len(sent) +1 ) ] for z in range(len(sent) +1)]
a=0
for i in tqdm(range(len(sent))):
    l=sent[i]
    pks=set()
    for A in tqdm(Unary.keys()):
        for pk in prob.keys():    
            if pk.split(' -> ')[1] in sent:
                pks.add(pk.split(' -> ')[0])
        if str(A+" -> "+l) in prob.keys():
            score[i][i+1][lhs_dict[A]] = float(prob[str(A+' -> '+l)])
            a+=1
    #print(pks)
    added=1
    print("started")
    while added:
        added=0
        for k in tqdm(Unary.keys()):
            for item in Unary[k]:
                if item in lhs_dict.keys():
                    if score[i][i+1][lhs_dict[item]]>0:
                        p1=prob[str(k)+' -> '+item]*score[i][i+1][lhs_dict[item]]
                        if p1>score[i][i+1][lhs_dict[k]]:
                            score[i][i+1][lhs_dict[k]]=p1
                            back[i][i+1][lhs_dict[k]]=item
                            print(back[i][i+1][lhs_dict[k]])
                            added=1


# In[ ]:


score[5][6][lhs_dict['NNS']]


# In[ ]:


for i in range(7):
    print([j for j, e in enumerate(score[i][6]) if e != 0])


# In[ ]:


Binary['S']
lhs_dict['S']


# In[ ]:


print(len(sent))
for span in tqdm(range(2,len(sent)+1)):
    print("span:"+str(span))
    for begin in tqdm(range(len(sent)-span+1)):
        end=begin+span
        print("span: "+str(span)+" begin: "+str(begin)+" end: "+str(end))
        for split in tqdm(range(begin+1,end)):
            print("span: "+str(span)+" begin: "+str(begin)+" end: "+str(end)+" split: "+str(split))
            print("Check in grammar")
            for k in Binary.keys():
                for item in Binary[k]:
                    if begin==0 and end==6:
                        if item[0] in lhs_dict.keys() and item[1] in lhs_dict.keys():
                            p1=score[begin][split][lhs_dict[item[0]]]*score[split][end][lhs_dict[item[1]]]*prob[str(k)+' -> '+item[0]+" "+item[1]]
                            print(score[begin][split][lhs_dict[item[0]]],score[split][end][lhs_dict[item[1]]])
                            if p1>score[begin][end][lhs_dict[A]]:
                                print(p1)
                                score[begin][end][lhs_dict[A]]=p1
                                back[begin][end][lhs_dict[A]]=(split,item[0],item[1])
                                print(back[begin][end][lhs_dict[A]])
                                print("S: "+k)
                                print(begin,end)
        added=1
        while added:
            added=0
            for k in tqdm(Unary.keys()):
                for item in Unary[k]:
                    if item in lhs_dict.keys():
                        if score[begin][end][lhs_dict[item]]>0:
                            p1=prob[str(k)+' -> '+item]*score[begin][end][lhs_dict[item]]
                            print(p1)
                            if p1>score[begin][end][lhs_dict[k]]:
                                score[begin][end][lhs_dict[k]]=p1
                                back[begin][end][lhs_dict[k]]=item
                                added=1


# In[ ]:


class GrammarTree(object):
    '''
    Tree data structure used to represent the grammar tree output generated by the cky algorithm
    '''
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

    def insertLeft(self, new_node):
            self.left = new_node

    def insertRight(self, new_node):
            self.right = new_node


# In[ ]:


from collections import deque
def print_level_order(head, queue = deque()):
    if isinstance(head,str):
        print(head)
        return
    print(head.data)
    [queue.append(node) for node in [head.left, head.right] if node]
    if queue:
        print_level_order(queue.popleft(), queue)


# In[ ]:


def build_tree(start,end,idx,back,non_terms):
    '''
    build_tree() builds tree from the backpointer matrix obtained in the cky() function
    :param start: start index for tree
    :param end: end index for tree
    :param idx: index used to find non_terminal
    :param back: the backpointer matrix
    :param non_terms: a list of non-terminals
    :return:
    '''
    tree = GrammarTree(non_terms[idx])
    node = back[start][end][idx]
    if isinstance(node,tuple):
        split,left_rule,right_rule = node
        tree.insertLeft(build_tree(start,split,left_rule,back,non_terms))
        tree.insertRight(build_tree(split,end,right_rule,back,non_terms))
        return tree
"""
    else:
        if node>0:
            tree.insertLeft(GrammarTree(non_terms[node]))
        return tree
"""


# In[ ]:


def get_parse_tree(score, back,non_terms):
    '''
    get_parse_tree() calls the build_tree() method
    :param score: score matrix
    :param back: backpointer matrix
    :param non_terms: list of non_terminals
    :return: GrammarTree the final parse tree
    '''
    root_index = score[0][len(score)-1].index(max(score[0][len(score)-1]))
    print(root_index)
    print(score[0][len(score)-1][root_index])
    tree = build_tree(0,len(score)-1,root_index,back,non_terms)
    print(tree)
    return tree


# In[ ]:


tree=get_parse_tree(score,back,reverse_dict)


# In[ ]:


reverse_dict[0]

