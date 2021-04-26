import random as rnd
import math



def sample_once(Q, num_iter, T0):
    keys = list(Q.keys())
    nodes = []
    for key in keys:
        un_found_0 = True
        un_found_1 = True
        for node in nodes:
            if(node == key[0]):
                un_found_0 = False
        if un_found_0:
            nodes.append(key[0])
        for node in nodes:
            if(node == key[1]):
                un_found_1 = False
        if un_found_1:
            nodes.append(key[1])
    # инициализация начального состояния
    position = {}
    for node in nodes:
        curr = 0
        if rnd.random() > 0.5:
            curr = 1
        position[node] = curr
    # побежали. Бежим num_iter*количество узлов
    for i in range(num_iter):
        position = consistent_change(nodes, Q, position, (T0/(i+1)))
    # расчитываем энергию и добавляем её
    f_en = full_energy(nodes, Q, position)
    position["energy"] = f_en
    # возвращаем
    return position


def stohastic_change(nodes, Q, curr_position, beta):
    for j in range(len(nodes)):
        i = rnd.randint(0, max(len(nodes)-1,0))
        delta_e = energy(nodes[i], nodes, Q, curr_position, 1-(curr_position[nodes[i]])) - \
                  energy(nodes[i], nodes, Q, curr_position, curr_position[nodes[i]])
        if (delta_e <= 0):
            curr_position[nodes[i]] = 1-(curr_position[nodes[i]])
        else:
            if rnd.random() < math.exp((-delta_e) / (beta)):
                curr_position[nodes[i]] = 1-(curr_position[nodes[i]])
    return curr_position


def consistent_change(nodes, Q, curr_position, beta):
    for i in range(len(nodes)):
        delta_e = energy(nodes[i], nodes, Q, curr_position, 1-(curr_position[nodes[i]])) - \
                  energy(nodes[i], nodes, Q, curr_position, curr_position[nodes[i]])
        if (delta_e <= 0):
            curr_position[nodes[i]] = 1-(curr_position[nodes[i]])
        else:
            if rnd.random() < math.exp((-delta_e) / (beta)):
                curr_position[nodes[i]] = 1-curr_position[nodes[i]]
    return curr_position




def energy(node, curr_nodes, Q, position, node_pos):
    en = 0
    for node1 in curr_nodes:
        if(node1==node):
            first_node = node
            second_node = node1
            en += Q[(first_node, second_node)] * node_pos * node_pos
        else:
            first_node = node
            second_node = node1
            en += Q[(first_node, second_node)] * node_pos * position[node1]


    return en


def full_energy(nodes, Q, position):
    en = 0
    for i in range(len(nodes)):
        for j in range(i, len(nodes)):
            en += Q[(nodes[i], nodes[j])]*position[nodes[i]]*position[nodes[j]]
    return en


def comparator(Q1,Q2):
    same = True
    for key in list(Q1.keys()):
        if(Q1[key]!=Q2[key]):
            same = False
    return same