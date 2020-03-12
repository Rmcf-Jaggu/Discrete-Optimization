#!/usr/bin/python
# -*- coding: utf-8 -*-

import time
import math
import random
import pdb

from collections import namedtuple
Item = namedtuple("Item", ['index', 'value', 'weight'])

global GLOBAL_VAR

def solve_it(input_data):
    global GLOBAL_VAR

    lines = input_data.split('\n')
    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count + 1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i - 1, int(parts[0]), int(parts[1])))

    GLOBAL_VAR = module(items)
    opt = 1 
    if len(items) <= 200:
        value, taken, tab = solve_it_pattern_dp(capacity, items)
    elif len(items) <= 400:
        value, taken, tab = solve_it_pattern_2(capacity, items, eps=0.01)
    elif len(items) <= 1000:
        value, taken, tab = solve_it_pattern_dp(capacity, items)
    else:
        opt = 0
        value, taken, visited = DFSearch(capacity, items)

    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(opt) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data

def module(items):
    ratio = [(0, 0)] * len(items)
    for trn in range(len(items)):
        ratio[trn] = (items[trn].value /
                     items[trn].weight, trn)
    ratio.sort(key=lambda x: x[0])
    return ratio

def solve_it_greedy(K, items, ordering='ratio'):
    variables = imp_elm(items, k=ordering)
    sol = [0] * len(items)
    v = 0
    w = 0
    for r in variables:
        if w + items[r[1]].weight > K:
            break
        v += items[r[1]].value
        w += items[r[1]].weight
        sol[r[1]] = 1
    generic_method(K, items, v, sol)
    return v, sol


def imp_elm(items, k='ratio'):
    tmp = [(0, 0)] * len(items)
    if k == 'ratio':
        for trn in range(len(items)):
            tmp[trn] = (items[trn].value / items[trn].weight, trn)
        tmp.sort(key=lambda x: x[0])
    elif k == 'value':
        for trn in range(len(items)):
            tmp[trn] = (items[trn].value, trn)
        tmp.sort(key=lambda x: x[0], reverse=True)
    elif k == 'size':
        for trn in range(len(items)):
            tmp[trn] = (items[trn].weight, trn)
        tmp.sort(key=lambda x: x[0])
    return tmp


def solve_it_pattern_dp(K, items):
    N = len(items)
    tab = [[0 for i in range(N + 1)] for j in range(K + 1)]

    for i in range(1, K + 1):
        for j in range(1, N + 1):
            if items[j - 1].weight > i:
                tab[i][j] = tab[i][j - 1]
            else:
                tab[i][j] = max(
                    tab[i][j - 1], tab[i - items[j - 1].weight][j - 1] + items[j - 1].value)

    opt = tab[K][N]
    taken = [0] * N
    i, b = (K, N)
    while b >= 1:
        if tab[i][b] == tab[i][b - 1]:
            taken[b - 1] = 0
        else:
            taken[b - 1] = 1
            i -= items[b - 1].weight
        b -= 1
    generic_method(K, items, opt, taken)
    return opt, taken, tab


def solve_it_pattern_2(K, rep, eps=0.2):
    N = len(rep)
    Lbound = max([i.value for i in rep])
    items = []
    for i in rep:
        items.append(Item(i.index, math.ceil(
            i.value / ((eps / N) * Lbound)), i.weight))
    P = sum([i.value for i in items])
    tab = [[max(P, K + 1) for p in range(P + 1)] for i in range(N + 1)]
    tab[0][0] = 0
    for i in range(1, N + 1):
        tab[i][0] = 0
    for i in range(1, N + 1):
        for j in range(1, P + 1):
            if items[i - 1].value <= j:
                tab[i][j] = min(tab[i - 1][j], items[i - 1].weight +
                                tab[i - 1][j - items[i - 1].value])
            else:
                tab[i][j] = tab[i - 1][j]
    opt = -1
    for p in range(P):
        if tab[N][p] <= K:
            opt = max(opt, p)
    sol = [0] * len(items)
    p = opt
    for i in range(N, 0, -1):
        if items[i - 1].value <= p:
            if items[i - 1].weight + tab[i - 1][p - items[i - 1].value] < tab[i - 1][p]:
                sol[i - 1] = 1
                p -= items[i - 1].value
    opt = 0
    for i in range(N):
        if sol[i] == 1:
            opt += rep[i].value
    generic_method(K, rep, opt, sol)
    return opt, sol, tab


def DFSearch(K, items):
    h = Node(K, items, [])
    rep = sorted(items, key=lambda x: x.value/x.weight, reverse=True)
    best_node, visited = def_find(K, rep, [h])
    sol = [0] * len(best_node.sol)
    for i in range(len(best_node.sol)):
        sol[rep[i].index] = best_node.sol[i]
    generic_method(K, items, best_node.value, sol)
    return best_node.value, sol, visited


def def_find(K, items, node_list):
    apx = None
    visited = 0
    time_limit = 300
    start_time = time.time()
    while len(node_list) > 0:
        node = node_list.pop()
        if node.feasible:
            if not node.is_leaf:
                if apx is None or node.estimate > apx.value:
                    def_visit(node, node_list, K, items)
                    visited += 1
            else:
                if apx is None or node.value > apx.value:
                    apx = node
        if time.time() - start_time > time_limit:
            print("Time interruption")
            break
    return apx, visited


def def_visit(node, node_list, K, items):
    sol1 = node.sol.copy()
    sol_l = sol1 + [1]
    sol_r = sol1 + [0]
    left = Node(K, items, sol_l)
    right = Node(K, items, sol_r)
    solve_bnb_depthfirst_method(node_list, left, right)


def solve_bnb_depthfirst_method(node_list, left, right):
    if right.feasible:
        node_list.append(right)
    if left.feasible:
        node_list.append(left)


def solve_bnb_bestfirst_method(node_list, left, right):
    if right.feasible:
        node_list.append(right)
    if left.feasible:
        node_list.append(left)
    if left.feasible or right.feasible:
        node_list.sort(key=lambda x: x.value)


def solve_random(node_list, left, right):
    r = random.random()
    if r > 0.5:
        l = [left, right]
    else:
        l = [right, left]
    if l[0].feasible:
        node_list.append(l[0])
    if l[1].feasible:
        node_list.append(l[1])


class Node():
    def __init__(self, K, items, sol):
        w, v = (0, 0)
        self.is_leaf = (len(sol) == len(items))
        for n in range(len(sol)):
            if sol[n]:
                w += items[n].weight
                v += items[n].value
        if w > K:
            self.feasible = False
        else:
            self.value = v
            self.feasible = True
            self.sol = sol
        if self.feasible:
            if not self.is_leaf:
                self.estimate = solve_it_relaxation(K, items, sol, self.value)
            else:
                self.estimate = self.value


def solve_it_relaxation(K, items, upd_sl, apx_value):
    global GLOBAL_VAR
    ratio = GLOBAL_VAR[len(upd_sl):len(items)]
    v = apx_value
    w = 0
    for i in range(len(upd_sl)):
        if upd_sl[i] == 1:
            w += items[i].weight
    for r in ratio:
        w += items[r[1]].weight
        if w <= K:
            v += items[r[1]].value
        else:
            extra = K -w
            perc = 1 - extra/items[r[1]].weight
            v += perc * items[r[1]].value
            return v
    return v


def generic_method(K, items, fin_value, sol):
    v = 0
    w = 0
    for i in range(len(sol)):
        if sol[i] == 1:
            v += items[i].value
            w += items[i].weight
    assert(v == fin_value)
    assert(w <= K)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')
