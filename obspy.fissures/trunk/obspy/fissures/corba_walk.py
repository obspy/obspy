"""
Walk throught CORBA object

SRC: http://www.bioinformatics.org/pipermail/pipet-devel/2000-March/001317.html
"""

import CosNaming

def name_to_string(name):
    """Convert CosNaming Name to a list of (id, kind) tuples."""
    res = []
    for i in range(len(name)):
        binding_name = name[i]
        res = res + [(binding_name.id, binding_name.kind)]
    return `res`

def walk(sofar, nameservice, visitor):
    """Walk the naming tree. Call visitor at object nodes."""
    (bl, bi) = nameservice.list(10000)
    print "%d bindings at naming context %s" % (len(bl), name_to_string(sofar))
    print "binding iterator: " + `bi`
    for i in range(len(bl)):
        binding = bl[i]
        try:
            obj = nameservice.resolve(binding.binding_name)
        except CosNaming.NamingContext.NotFound:
            print "Could not find %s" % name_to_string(sofar + binding.binding_name)
            continue

        if binding.binding_type is CosNaming.ncontext:
            walk(sofar + binding.binding_name, obj, visitor)
        else:
            visitor(sofar, binding, obj)

def walk_print(nameservice):
    """Walk the naming tree. Print object nodes."""
    def printit(sofar, binding, obj):
        """Print object node."""
        print name_to_string(sofar + binding.binding_name)
        print obj

    walk([], nameservice, printit)
