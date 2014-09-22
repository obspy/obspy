class SeismicPhase(object):
    """Stores and transforms seismic phase names to and from their corresponding sequence of branches.
    Will maybe contain "expert" mode wherein paths may start in the core. Principal use is to calculate leg
    contributions for scattered phases. Nomenclature: "K" - downgoing wave from source in core;
     "k" - upgoing wave from source in core.
    """

    def __init__(self, name, tMod):
        self.name = name
        self.sourceDepth = tMod.sourceDepth
        self.tMod = tMod
        self.legs = self.legPuller(name)
        self.createPuristName(tMod)
        self.parseName(tMod)
        self.sumBranches(tMod)

    # Java static method, so I think that means:
    @classmethod
    def legPuller(cls, name):
        """Are you pulling my leg?"""
        pass

    def createPuristName(self, tMod):
        pass

    def parseName(self, tMod):
        pass

    def sumBranches(self, tMod):
        pass