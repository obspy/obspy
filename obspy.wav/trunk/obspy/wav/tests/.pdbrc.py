# http://code.activestate.com/recipes/498182/
# save this in .pdbrc.py in your home directory
def complete(self, text, state):
    """return the next possible completion for text, using the current frame's local namespace

       This is called successively with state == 0, 1, 2, ... until it
       returns None.  The completion should begin with 'text'.
    """
    import rlcompleter
    # keep a completer class, and make sure that it uses the current local scope
    if not hasattr(self, 'completer'):
        self.completer = rlcompleter.Completer(self.curframe.f_locals)
    else:
        self.completer.namespace = self.curframe.f_locals
    return self.completer.complete(text, state)
