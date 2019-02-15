from obspy import read_inventory

#def test_wildcard():
inv = read_inventory('a*.xml')
print(len(inv))
assert len(inv) == 2
assert inv[0].code == 'BW'
assert inv[1].code == 'GR'
