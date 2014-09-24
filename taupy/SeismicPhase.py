from .helper_classes import TauModelError


class SeismicPhase(object):
    """Stores and transforms seismic phase names to and from their corresponding sequence of branches.
    Will maybe contain "expert" mode wherein paths may start in the core. Principal use is to calculate leg
    contributions for scattered phases. Nomenclature: "K" - downgoing wave from source in core;
     "k" - upgoing wave from source in core.
    """

    def __init__(self, name, tMod):
        #The phase name, e.g. PKiKP.
        self.name = name
        self.sourceDepth = tMod.sourceDepth
        self.tMod = tMod
        self.legs = legPuller(name)
        # Name with depths corrected to be actual discontinuities in the model.
        self.puristName = self.createPuristName(tMod)
        self.parseName(tMod)
        self.sumBranches(tMod)

    def createPuristName(self, tMod):
        currLeg = self.legs[0]
        # Deal with surface wave veocities first, since they are a special case.
        if len(self.legs) == 2 and currLeg.endswith("kmps"):
            puristName = self.name
            return puristName
        puristName = ""
        # Only loop to penultimate element as last leg is always "END".
        for currLeg in self.legs[:-1]:
            # Find out if the next leg represents a phase conversion or reflection depth.
            if currLeg.startswith("v") or currLeg.startswith("^"):
                disconBranch = closestBranchToDepth(tMod, currLeg[1])
                legDepth = tMod.tauBranches[0][disconBranch].topDepth
                puristName += currLeg[0]
                puristName += str(legDepth)
            else:
                try:
                    float(currLeg)
                    # If it is indeed a number:
                    disconBranch = closestBranchToDepth(tMod, currLeg)
                    legDepth = tMod.tauBranches[0][disconBranch].topDepth
                    puristName += str(legDepth)
                except ValueError:
                    # If currLeg is just a string:
                    puristName += currLeg
        return puristName

    def parseName(self, tMod):
        pass

    def sumBranches(self, tMod):
        pass


def closestBranchToDepth(tMod, depthString):
    """Finds the closest discontinuity to the given depth that can hae reflections and phase transformations."""
    if depthString == "m":
        return tMod.mohoBranch
    elif depthString == "c":
        return tMod.cmbBranch
    elif depthString == "i":
        return tMod.iocbBranch
    # Non-standard boundary, given by a number: must look for it.
    disconBranch = -1
    disconMax = 1e300
    disconDepth = float(depthString)
    for i, tBranch in enumerate(tMod.tauBranches[0]):
        if (abs(disconDepth - tBranch.topDepth) < disconMax and not
             any(ndc == tBranch.topDepth for ndc in tMod.noDisconDepths)):
            disconBranch = i
            disconMax = abs(disconDepth - tBranch.topDepth)
    return disconBranch


def legPuller(name):
    """Tokenizes a phase name into legs, ie PcS becomes 'P'+'c'+'S' while p^410P
    would become 'p'+'^410'+'P'. Once a phase name has been broken into
    tokens we can begin to construct the sequence of branches to which it
    corresponds. Only minor error checking is done at this point, for
    instance PIP generates an exception but ^410 doesn't. It also appends
    "END" as the last leg."""
    # Java static method, so I think that means making it a function.
    # or @classmethod? But it doesn't need the class.
    offset = 0
    legs = []
    # Special case for surface wave velocity.
    if name.endswith("kmps"):
        legs.append(name)
    else:
        while offset < len(name):
            nchar = name[offset]
            # Do the easy ones, i.e. K, k, I, i, J, p, s, m, c:
            if any(nchar == c for c in ("K", "k", "I", "i", "J", "p", "s", "m", "c")):
                legs.append(nchar)
                offset += 1
            elif nchar == "P" or "S":
                # Now it gets complicated, first see if the next char is part of a different leg or if it's the end.
                if (offset + 1 == len(name) or any(name[offset + 1] == c for c in ("P", "S", "K", "m", "c", "^", "v"))
                     or name[offset + 1].isdigit()):
                    legs.append(nchar)
                    offset += 1
                elif name[offset + 1] == "p" or name[offset + 1] == "s":
                    raise TauModelError("Invalid phase name: \n "
                                        "{} cannot be followed by {} in {}.".format(nchar, name[offset+1], name))
                elif any(name[offset+1] == c for c in ("g", "b", "n")):
                    # The leg is not described by one letter, check for two:
                    legs.append(name[offset:offset+2])
                    offset += 2
                elif len(name) >= offset + 5 and any(name[offset:offset+5] == c for c in ("Sdiff", "Pdiff")):
                    legs.append(name[offset:offset+5])
                    offset += 5
                else:
                    raise TauModelError("Invalid phase name: \n "
                                        "{nchar} in {name}".format(**locals()))
            elif nchar == "^" or nchar == "v":
                # Top side or bottom side reflections, check for standard boundaries and then check for numerical ones.
                if any(name[offset+1] == c for c in ("m", "c", "i")):
                    legs.append(name[offset:offset+2])
                    offset += 2
                elif name[offset+1].isdigit() or name[offset+1] == ".":
                    numString = name[offset]
                    offset += 1
                    while name[offset+1].isdigit() or name[offset+1] == ".":
                        numString += name[offset]
                        offset += 1
                    legs.append(numString)
                else:
                    raise TauModelError("Invalid phase name {nchar} in {name}.".format(**locals()))
            elif nchar.isdigit() or nchar == ".":
                numString = name[offset]
                offset += 1
                while name[offset+1].isdigit() or name[offset+1] == ".":
                    numString += name[offset]
                    offset += 1
                legs.append(numString)
            else:
                raise TauModelError("Invalid phase name {nchar} in {name}.".format(**locals()))
    legs.append("END")
    phaseValidate(legs)
    return legs


def phaseValidate(legs):
    # Raise an exception here if validation fails.
    # Todo: implement phase names validation (maybe not so necessary...)
    pass
