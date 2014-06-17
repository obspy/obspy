class testclass(object):
    defaultno = 5
    
    def __init__(self, no = defaultno):
        self.no = no
        
    def meep(self):
        print(self.no)
        
    # The classmethod decorator is necessary, else the cls isn't 
    # recognised (gives an error that it's undefined)
    @classmethod
    def classmeep(cls):
        #these are both invalid:
        #print(no)
        #print(self.no)
        print(cls.defaultno)
        
        
testclass.classmeep()
test = testclass(3)
test.meep()

# NB that you could rename defaultno to no as well and get same result - I suppose that's because the namespaces are separate.
