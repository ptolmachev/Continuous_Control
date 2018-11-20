import numpy as np
import random

class SumTree():
    def __init__(self, capacity):
        self.capacity = capacity #maximum number of stored elements
        self.tree = np.zeros( 2*capacity - 1 )
        self.data = np.full((capacity,), None) #np.zeros(capacity, dtype = object)#
        self.len = 0
        self.write = 0
        self.full = 0
        
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        if p <=0:
            raise ValueError()
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.len += 1
        self.write += 1
        if self.write >= self.capacity:
            if self.full == 0:
                self.full = 1
                print("\nBuffer is full")
            self.write = 0
            self.len = self.capacity

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def update_multiple(self, idxs, ps):
        if len(ps) != len(idxs):
            raise IndexError()
        for i in range(len(idxs)):
            self.update(idxs[i], ps[i])

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])
    
    def sample(self, quantity):
        if quantity >= self.len:
            quantity = self.len-1
        idxs = []
        idxs_tree = []
        list_of_sampled_items = []
        #while len(list_of_sampled_items) < quantity:
        for i in range(quantity):
            s = np.random.rand()*self.total()
            indexed_item = self.get(s)

            # if (indexed_item[0] not in idxs) and (indexed_item[2] is not None):
            if (indexed_item[2] is not None):
                idxs.append(indexed_item[0])
                list_of_sampled_items.append(indexed_item[2])

        if len(idxs) != len(list_of_sampled_items):
            raise IndexError(len(idxs), len(list_of_sampled_items))
        return idxs, list_of_sampled_items

    def get_last_k_elements(self,k):
        idxs = []
        list_of_last_k_elements = []
        for i in range(k):
            ind = (self.write - i)%self.capacity + self.capacity - 1
            idxs.append(ind)
            dataIdx = ind - self.capacity + 1
            list_of_last_k_elements.append(self.data[dataIdx])
        return idxs, list_of_last_k_elements


    def __len__(self):
        return self.len