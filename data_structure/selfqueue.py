class SelfQueue:
    def __init__(self):
        self.items = []

    def add(self, item):
        self.items.append(item)

    def get(self):
        return self.items[0]

    def put(self):
        return self.items.pop(0)

    def clear(self):
        del self.items[:]

    def empty(self):
        return self.size() == 0

    def reverse(self):
        self.items.reverse()

    def size(self):
        return len(self.items)
