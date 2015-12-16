# make list of imports that should work
python - <<-EOF
from obspy.core.util.deprecation_helpers import FUNCTION_MAPS
with open("/tmp/obspy_reroute_list.txt", "wb") as fh:
    for mod, mapping in FUNCTION_MAPS.items():
        for old_name in mapping.keys():
            fh.write('{};{}\n'.format(mod, old_name))
EOF
# try these imports, one by one in a fresh python interpreter
for MODFUNC in `cat /tmp/obspy_reroute_list.txt`
do
MODFUNC=(${MODFUNC//;/ })
MOD=${MODFUNC[0]}
FUNC=${MODFUNC[1]}
python - <<-EOF
import warnings
warnings.filterwarnings("ignore")
import obspy
try:
    from $MOD import $FUNC
except ImportError:
    print("FAILED: $MOD $FUNC")
else:
    print("OK    : $MOD $FUNC")
EOF
done
