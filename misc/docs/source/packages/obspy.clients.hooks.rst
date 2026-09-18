.. currentmodule:: obspy.clients.hooks
.. automodule:: obspy.clients.hooks

    .. comment to end block

    The Hook Protocol
    ------------------
    The interface a request hook is written against: the request view it
    receives, and the error it may raise.

    .. autosummary::
       :toctree: autogen
       :nosignatures:

       HookRequest
       RequestHookError
       RequestHookHandler

    .. comment to end block

    Included Hooks
    ---------------
    Ready-made hooks for common cases. Each is just a callable satisfying
    the protocol above - a hook you write yourself works everywhere these
    do.

    .. autosummary::
       :toctree: autogen
       :nosignatures:

       BearerTokenHook
       LoggingHook
       chain

    .. comment to end block
