
# coding: utf-8

# In[1]:


import nltk
from nltk import Tree
from nltk.corpus import treebank
from nltk import Nonterminal, Production
from nltk import induce_pcfg


# In[8]:


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


# In[9]:


Leaves=list(Leaves)


# In[11]:


from nltk  import Nonterminal
S = Nonterminal('S')
grammar = induce_pcfg(S, productions)


# In[14]:


from nltk.parse import ViterbiParser
from functools import reduce
sentence="I saw it"
tokens = sentence.split(' ')
parser = ViterbiParser(grammar)
parses = parser.parse_all(tokens)

if parses: 
    lp = len(parses)
    p = reduce(lambda a,b:a+b.prob(), parses, 0.0)
else:
    p = 0
for parse in parses:
    print(parse)


# In[ ]:


count = {}
rule_count = {}
rule_probs = {}
for rule in grammar:
    if rule[0] not in count:
        count[rule[0]] = 1
    else:
        count[rule[0]] += 1
for rule in grammar:
    if rule not in rule_count:
        rule_count[rule] = 1
    else:
        rule_count[rule] += 1
for entry in rule_count:
    rule_probs[entry] = float(rule_count[entry])/count[entry[0]]


# In[14]:


grammar_rules=[]
for line in productions:
    grammar_rules.append(tuple(str(line).rstrip('\n').split(' -> ')))


# In[15]:


count = {}
rule_count = {}
rule_probs = {}
for rule in grammar_rules:
    if rule[0] not in count:
        count[rule[0]] = 1
    else:
        count[rule[0]] += 1
for rule in grammar_rules:
    if rule not in rule_count:
        rule_count[rule] = 1
    else:
        rule_count[rule] += 1
for entry in rule_count:
    rule_probs[entry] = float(rule_count[entry])/count[entry[0]]


# In[20]:


non_terms = set()
for rules in set(grammar_rules):
    non_terms.add(rules[0])
non_terms=list(non_terms)
sent="I saw John with my eyes"
sent=sent.split(" ")
print(sent)


# In[22]:


score=[[[0.0 for i in range(len(non_terms))] for j in range(len(sent)+1)] for k in range(len(sent)+1)]
back =[[[-1 for i in range(len(non_terms))] for j in range(len(sent)+1)] for k in range(len(sent)+1)]


# In[32]:


rule_index = {}
from tqdm import tqdm_notebook as tqdm
for i,word in tqdm(enumerate(sent)):
    rules_used = []
    rules_not_used = []
    for j,A in tqdm(enumerate(non_terms)):
        r = A, '\'' + word + '\''
        if r in grammar_rules:
            score[i][i+1][j] = rule_probs[r]
            rules_used.append(j)
            rule_index[A] = j
        else:
            rules_not_used.append(j)
            rule_index[A] = j
    rules_used_temp = rules_used[:]
    rules_not_used_temp = rules_not_used[:]
    added = True
    while added:
        print(added)
        added = False
        for a in rules_not_used:
            for b in rules_used:
                r = non_terms[a], non_terms[b]
                if r in grammar_rules:
                    prob = rule_probs[r] * score[i][i+1][b]
                    if prob > score[i][i+1][a]:
                        score[i][i+1][a] = prob
                        back[i][i+1][a] = b
                        rules_used_temp.append(a)
                        try:
                            rules_not_used_temp.remove(a)
                            rules_used_temp.remove(b)
                        except ValueError:
                            pass
                        added = True

        rules_used = rules_used_temp[:]
        rules_not_used =rules_not_used_temp[:]

    


# In[34]:


bin_set = set()
for rules in grammar_rules:
    if len(rules[1].split(' ')) == 2:
        b, c = rules[1].split(' ')
        bin_set.add((rules[0],b,c))
        bin_set.add((rules[0], c, b))


# In[ ]:


binary_rules =list(bin_set)
for span in tqdm(range(2,len(sent)+1)):
    print("span:"+str(span))
    for begin in tqdm(range(len(sent)+1-span)):
        rules_used = []
        end = begin + span
        print("span: "+str(span)+" begin: "+str(begin)+" end: "+str(end))
        for split in tqdm(range(begin+1, end)):
            print("span: "+str(span)+" begin: "+str(begin)+" end: "+str(end)+" split: "+str(split))
            for rule in tqdm(binary_rules):
                a, b, c = rule_index[rule[0]], rule_index[rule[1]], rule_index[rule[2]]
                concat_rule = rule[0], ' '.join((rule[1], rule[2]))
                if concat_rule in grammar_rules:
                    prob = score[begin][split][b] * score[split][end][c] * rule_probs[concat_rule]
                else:
                    continue
                if prob > score[begin][end][a]:
                    score[begin][end][a] = prob
                    back[begin][end][a] = split, b, c
                    rules_used.append(a)

            ### Handle Unaries
        added = True
        while added:
            added = False
            for a in range(len(non_terms)):
                for b in rules_used:
                    r = non_terms[a], non_terms[b]
                    if r in grammar_rules:
                        prob = rule_probs[r] * score[begin][end][b]
                        if prob > score[begin][end][a]:
                            score[begin][end][a] = prob
                            back[begin][end][a] = b
                            added = True


# In[26]:


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
    print(A)
    BC=str(grammar.productions()[i]).split(' -> ')[1].split()[:-1]
    source[A]=BC
    lhs_1.add(A)
    total.add(A)
    p1=float(str(grammar.productions()[i]).split(' -> ')[1].split()[-1][1:-1])
    if len(BC)==1:
        prob[str(grammar.productions()[i]).split(' -> ')[0]+" -> "+str(BC[0])]=p1
        rhs_1.add(BC[0])
        total.add(BC[0])
    elif len(BC)==2:
        prob[str(grammar.productions()[i]).split(' -> ')[0]+" -> "+str(BC[0])+" "+str(BC[1])]=p1
        rhs_1.add(BC[0])
        rhs_1.add(BC[1])
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


# In[6]:


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


# In[7]:


Binary


# In[8]:


total_dict['VP|<VP-PP>']
total1=list(total)


# In[27]:


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
            print(A)
            score[i][i+1][lhs_dict[A]] = float(prob[str(A+' -> '+l)])
            print(score[i][i+1][lhs_dict[A]],i)
            a+=1
    print(pks)
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


# In[31]:


for i in range(7):
    print([j for j, e in enumerate(score[i][6]) if e != 0])


# In[28]:


Binary['S']
lhs_dict['S']


# In[29]:


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
                            if score[begin][split][lhs_dict[item[0]]]!=0 and score[split][end][lhs_dict[item[1]]]: 
                                print(score[begin][split][lhs_dict[item[0]]])
                                print(score[split][end][lhs_dict[item[1]]])
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


# In[13]:


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


# In[14]:


from collections import deque
def print_level_order(head, queue = deque()):
    if isinstance(head,str):
        print(head)
        return
    print(head.data)
    [queue.append(node) for node in [head.left, head.right] if node]
    if queue:
        print_level_order(queue.popleft(), queue)


# In[15]:


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


# In[16]:


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


# In[30]:


tree=get_parse_tree(score,back,reverse_dict)


# In[18]:


reverse_dict[0]

