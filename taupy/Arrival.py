class Arrival:
    """
    Convenience class for storing the parameters associated with a phase arrival.
    """
    def __init__(self, phase, time, dist, rayParam, rayParamIndex,
                 name, puristName, sourceDepth, takeoffAngle, incidentAngle):
        # phase that generated this arrival
        self.phase = phase
        # travel time in seconds
        self.time = time
        # angular distance (great circle) in radians
        self.dist = dist
        # ray parameter in seconds per radians
        self.rayParam = rayParam
        self.rayParamIndex = rayParamIndex
        # phase name
        self.name = name
        # phase name changed for true depths
        self.puristName = puristName
        # source depth in kilometers
        self.sourceDepth = sourceDepth
        self.incidentAngle = incidentAngle
        self.takeoffAngle = takeoffAngle
        # pierce and path points
        self.pierce, self.path = [], []

    def getPierce(self):
        """
        Returns pierce points as TimeDist objects.
        """
        if not self.pierce:
            self.pierce == self.phase.calcPierce(self).getPierce()
        return self.pierce

    def getPath(self):
        """
        Returns pierce points as TimeDist objects.
        """
        if not self.path:
            self.path == self.phase.calcPath(self).getPath()
        return self.path