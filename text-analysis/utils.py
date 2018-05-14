class Incrementer:
    #create and incrementer
    def __init__(self, item):
        self.position = 0
        self.item = item

    def set(self):
        try:
            n =self.item[self.position]
            return n
        except Exception as e:
            print(e)
            print(self.item)
            print(self.position)
            print(len(self.item))


    def increment(self):
        self.position += 1
        return self.set()

    def has_next(self):
        # print("inside has next")
        # print("position", self.position)
        # print("length list", len(self.item))
        if self.position == len(self.item) - 1:

            return False

        else:
            return True