from .helper_classes import TauModelError


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
        self.legs = legPuller(name)
        self.createPuristName(tMod)
        self.parseName(tMod)
        self.sumBranches(tMod)

    def createPuristName(self, tMod):
        pass

    def parseName(self, tMod):
        pass

    def sumBranches(self, tMod):
        pass


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
    pass
