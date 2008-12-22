#!/usr/bin/env python


import sacio as p

fn = './LWTT.BHN.SAC'
t = p.ReadSac()
ok = t.ReadSacFile(fn)
t.ListStdValues()
