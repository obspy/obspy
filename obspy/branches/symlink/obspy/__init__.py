from pkg_resources import EntryPoint, iter_entry_points, working_set, Environment
import pkg_resources, sys
from distutils.core import run_setup
# http://dev.lazaridis.com/base/browser/infra/tracx/loader-draft-test.py


#dist = run_setup('setup.py', None, 'init')
#import pdb; pdb.set_trace()

#ep_map = pkg_resources.EntryPoint.parse_map(dist.entry_points, dist)
ep_map = EntryPoint.parse_map(
"""
[obspy.plugin.waveform]
GSE2 = obspy.gse2.core

[obspy.plugin.waveform.GSE2]
isFormat = obspy.gse2.core:isGSE2
readFormat = obspy.gse2.core:readGSE2
writeFormat = obspy.gse2.core:writeGSE2
"""
)
#working_set.add_entry(".")
#env = Environment(["."])
#working_set.require(*env.keys())

    
for group in ep_map.keys():
    ep_list = ep_map[group]
    print ep_list
    
    for name, entry_point in ep_list.iteritems():
        print entry_point
        try:
            entry_point.load(require=False)
        except pkg_resources.DistributionNotFound, e:
            pass
sys.path.append( '.' )

print "\niter_entry_points\n"
for ep in iter_entry_points('obspy.plugin.waveform'):
    print ep
