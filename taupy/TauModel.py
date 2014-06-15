class TauModel(object):
    """Dummy class"""
    
    DEBUG = False

    def __init__(self, sMod):
        
        self.sMod = sMod
        print("This is the init method of TauModel clocking in. Hello!")
        print("The debug flag for TauModel is set to:"+str(self.DEBUG), 
              "the default here is false but it should have been modified TauP_Create.")

        
    def writeModel(self, outfile):
        with open(outfile, 'w') as f:
            f.write("A tau model should be here")

    def __str__(self):
        desc = 'This is the TauModel __str__ method.'
        return desc
