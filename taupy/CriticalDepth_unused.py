class CriticalDepth:
    
    ''' Utility class to keep track of criticalpoints (discontinuities or reversals
    in slowness gradient) within slowness and velocity models.
    
    /** depth in kilometers at which there is a critical point. */
    private double depth;

    /** layer number within the velocity model with this depth at its top. */
    private int velLayerNum;

     * slowness layer for P waves with this depth at its top. This can be
     * PLayers.size() for the last critical layer.
    private int PLayerNum;

     * slowness layer for S waves with this depth at its top. This can be
     * SLayers.size() for the last critical layer.
    private int SLayerNum;
    '''
    def __init__(self, depth, velLayerNum, pLayerNum, sLayerNum)
        self.depth = depth
        self.velLayerNum = velLayerNum;
        self.sLayerNum = pLayerNum;
        self.sLayerNum = pLayerNum;
