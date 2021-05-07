import numpy as np
import pandas as pd
from Men1.ClassicalAnnealingSimulation import *

N_levels = 10 # Количество состояний, выводимых при выводе sampleSet
N_levels_comparing = 5;

def creating_QUBO_simple(X_curr,y_curr,lambbda):
    Q = {}
    G = np.zeros((X_curr.shape[1], X_curr.shape[1]))
    k = np.ones(X_curr.shape[1], )*lambbda
    X_1 = X_curr
    for i in range(X_curr.shape[1]):
        curr = lambbda
        for j in range(X_curr.shape[0]):
            curr-=y_curr[j]*X_curr[j][i]
        k[i] = 2*curr
    for i in range(X_curr.shape[1]):
        for j in range(X_curr.shape[1]):
            curr = 0
            for l in range(X_curr.shape[0]):
                curr += X_curr[l][i]*X_curr[l][j]
            G[i][j] = curr

    for i in range(X_curr.shape[1]):
        name_1 = 'x' + str(i+1)
        Q[(name_1, name_1)] = G[i, i]+k[i]
        for j in range(i+1, X_curr.shape[1]):
            name_2 = 'x' + str(j+1)
            Q[(name_1, name_2)] = G[i, j] + G[j, i]
    return Q


def creating_QUBO(X_curr,y_curr,lambbda):
    Q = {}
    G = np.zeros((X_curr.shape[1], X_curr.shape[1]))
    k = np.ones(X_curr.shape[1], )*lambbda
    k = k - 2 * np.dot(y_curr.T, X_curr).T

    for i in range(X_curr.shape[0]):
        G = G + np.dot((X_curr[i]).reshape((X_curr.shape[1],1)), (X_curr[i]).reshape(1,X_curr.shape[1]))

    for i in range(X_curr.shape[1]):
        name_1 = 'x' + str(i+1)
        Q[(name_1, name_1)] = G[i, i]+k[i]
        for j in range(i+1, X_curr.shape[1]):
            name_2 = 'x' + str(j+1)
            Q[(name_1, name_2)] = G[i, j] + G[j, i]
    return Q


def binominalization_0001(X):
    # 0001 - A
    # 0010 - C
    # 0100 - G
    # 1000 - T
    X_b = []
    for i in range(len(X)):
        X_b.append([])
        for j in range(len(X[i])):
            if X[i][j] == "A":
                X_b[i].append(0)
                X_b[i].append(0)
                X_b[i].append(0)
                X_b[i].append(1)
            elif X[i][j] == "C":
                X_b[i].append(0)
                X_b[i].append(0)
                X_b[i].append(1)
                X_b[i].append(0)
            elif X[i][j] == "G":
                X_b[i].append(0)
                X_b[i].append(1)
                X_b[i].append(0)
                X_b[i].append(0)
            elif X[i][j] == "T":
                X_b[i].append(1)
                X_b[i].append(0)
                X_b[i].append(0)
                X_b[i].append(0)
    return np.array(X_b)


def binominalization_01(X):
    # 00 - A
    # 01 - C
    # 10 - G
    # 11 - T
    X_b = []
    for i in range(len(X)):
        X_b.append([])
        for j in range(len(X[i])):
            if X[i][j] == "A":
                X_b[i].append(0)
                X_b[i].append(0)
            elif X[i][j] == "C":
                X_b[i].append(0)
                X_b[i].append(1)
            elif X[i][j] == "G":
                X_b[i].append(1)
                X_b[i].append(0)
            elif X[i][j] == "T":
                X_b[i].append(1)
                X_b[i].append(1)
    return np.array(X_b)


def Temp_by_time(time):
    return 100.0 / (1.0*time+1.0)


class samplerSA:
    def __init__(self):
        pass

    def sample_qubo(self, Q, num_reads = 100, num_iter = 100, T0 = 30):
        results = []

        # дополняем матрицу связок до полной (это нужно что бы после избежать костылей)
        Keys0 = []
        Real_Keys = []

        for key in list(Q.keys()):
            Q[(key[1], key[0])] = Q[key]
            Keys0.append(key[0])
            Keys0.append(key[1])
        for i in range(len(Keys0)):
            un_found = True
            for j in range(len(Real_Keys)):
                if Real_Keys[j] == Keys0[i]:
                    un_found = False
            if un_found:
                Real_Keys.append(Keys0[i])

        for key1 in Real_Keys:
            for key2 in Real_Keys:
                if (key1, key2) in list(Q.keys()):
                    pass
                else:
                    Q[(key1, key2)] = 0


        num_res = []
        for i in range(num_reads):
            new_res = sample_once(Q, num_iter, lambda t: Temp_by_time(t))
            if(i%10==9):
                print("Запуск номер "+str(i+1))

            un_found = True
            for j in range(len(num_res)):
                if comparator(results[j], new_res):
                    un_found = False
                    num_res[j] += 1
            if un_found:
                num_res.append(1)
                results.append(new_res)
        df = pd.DataFrame()
        if(len(results)>0):
            # сортируем результаты по убыванию частоты
            for i in range(len(num_res)-1, 0, -1):
                for j in range(0,i-1):
                    if num_res[j] < num_res[j+1]:
                        results[j], results[j+1] = results[j+1], results[j]
                        num_res[j], num_res[j+1] = num_res[j+1], num_res[j]
            columns = list(results[0].keys())
            data = {}
            for col in columns:
                column_data = []
                for i in range(len(results)):
                    column_data.append(results[i][col])
                data[col] = column_data
            data['num_oc.'] = num_res
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame({})
        return sampleSet(df)


class sampleSet:
    def __init__(self, data):
        self.data = data
        self.num_iter = min(len(data), N_levels)
        self.columns = self.data.columns
        self.head = " "
        self.col_coords = []
        a = 1
        for i in range(len(self.columns)):
            self.head += " " + self.columns[i]
            a += 1 + len(self.columns[i])
            self.col_coords.append(a)
        self.w = self.data.to_numpy()
        self.w = ((self.data.to_numpy().T)[0:-2]).T

    def __str__(self):
        our_str = "";
        our_str+=self.head+"\n"
        for i in range(self.num_iter):
            line = ""
            line = str(i)
            for j in range(len(self.columns)):
                while(len(line)<self.col_coords[j]-len(str(self.data[self.columns[j]][i]))):
                    line += " "
                line += str(self.data[self.columns[j]][i])
            our_str+=line+"\n"
        return our_str


    def predict(self, X):
        y = [];
        for i in range(X.shape[0]):
            curr = 0.0
            for j in range(N_levels_comparing):
                curr += 1.0*np.dot(self.w[j], (X[i]).T)/(1.0*N_levels_comparing)
            y.append(curr)
        y = np.array(y)
        return y


def predictDW(ss, X):
    y = []
    A = ss.data()
    w = np.zeros((len(list(A)[0][0]),))
    for i in range(N_levels_comparing):
        for j in range(len(list(A)[0][0])):
            w[j] += list(A)[i][0]['x'+str(j+1)]
    w = w / (1.0*N_levels_comparing)
    return np.dot(X, (w.reshape((len(list(A)[0][0]), 1))))
