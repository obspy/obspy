#from VelocityLayer import *
from taupy.VelocityModel import *

test = VelocityModel()

print(test)

test2 = VelocityModel.readVelocityFile('iasp91.tvel')  # test_file.tvel is shorter

print(test2)

#for i, layer in enumerate(test2.layers):
#    print(layer)

print(validate())

print(len(test2))
print(test2.getNumLayers()) # probably ought to throw out that method...

print(test2.getDisconDepths())

print(test2.layerNumberAbove(30))
print(test2.layerNumberBelow(0))

#eval at cmb
print(test2.evaluateAbove(2889.0, 'p'))
print(test2.evaluateBelow(2889.0, 'D'))


print(test2.depthAtTop(50))
print(test2.depthAtBottom(50))

print(test2.mohoDepth, test2.iocbDepth, test2.cmbDepth)
print(test2.fixDisconDepths())
print(test2.mohoDepth, test2.iocbDepth, test2.cmbDepth)
