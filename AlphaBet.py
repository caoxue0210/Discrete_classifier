from collections import OrderedDict

class AlphaBet:
    def __init__(self):
        self.list = []
        self.dict = OrderedDict()
    # def makeVocab(self, inst):
    #     for i in inst:
    #         if i not in self.list:
    #             self.list.append(i)
    #             #print(self.list)
    #     for k in range(len(self.list)):
    #         self.dict[self.list[k]] = k
    #     return self.list
    def makeVocab(self, inst):
        for i in inst:
            self.list.append(i)
        for k in range(len(self.list)):
            self.dict[self.list[k]] = k
        return self.list
