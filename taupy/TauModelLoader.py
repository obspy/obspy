def internalLoad(modelName, searchPath, verbose):
    if modelName.endswith(".taup"):
        filename = modelName
    else:
        filename = modelName + ".taup"
    # Java tries to:
    # - find the model in the distributed taup.jar
    # - find it in the classpath (?)
    # - find it in the path specified by the taup.model.path property.
    #    This is what's passed in as searchPath!
    # - find it in the current directory
    # - try to load a velocity model of the same name and run TauP_Create.createTauModel
    #
    # I'll keep it simple for now; stick to the specified path.
    # I also don't think finding the model just somewhere in the paths is good practice.
    # Maybe put the following in a TauModel.readModel method?
    import pickle
    with open(filename, 'rb') as f:
        return pickle.load(f)


def load(modelName, searchPath, verbose):
    """Read a tau model that was previously saved."""
    out = loadFromCache(modelName)
    if out is None:
        out = internalLoad(modelName, searchPath, verbose)
        # cache(out)
    return out


def loadFromCache(modelName):
    """Caching could be useful for many reruns..."""
    # Todo: think about caching    
    return None
