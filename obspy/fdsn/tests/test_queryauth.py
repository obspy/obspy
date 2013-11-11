#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Document tests for queryauth

These tests exist in this form because digest authentication may not confirm
the authentication, even if credentials are valid, unless a specific
server:port are accessed.

"""

from obspy.fdsn import Client
from obspy import UTCDateTime
from obspy.fdsn.header import DEFAULT_USER_AGENT, FDSNException
import unittest
import os
import csv
# import urllib2

USER_AGENT = "ObsPy (test suite) " + " ".join(DEFAULT_USER_AGENT.split())


class QueryauthTestCase(unittest.TestCase):

    def setUp(self):
        # directory where the test files are located
        self.path = os.path.dirname(__file__)
        self.datapath = os.path.join(self.path, "data")
        self.client = Client(base_url="IRIS", user_agent=USER_AGENT)
        self.client_auth = \
            Client(base_url="IRIS", user_agent=USER_AGENT,
                   user="nobody@iris.edu", password="anonymous")
        self.client = Client("http://wslive6:8082",
                             debug=True, user="nobody@iris.edu",
                             password="anonymous")
        #print(self.client)

    def get_help_from_services(self):
        services_to_poke_at = ["dataselect", "bulkdataselect", "event",
                               "station", "eh", None]
        for svc in services_to_poke_at:
            try:
                self.client.help(svc)
            except:
                pass
    def set_up_waveform_tests(self):
        """
        load request parameters from a file, along with expected number of
        retrieved traces and the HTTP Code
        """

        params_and_result = []
        with open(os.path.join(self.datapath,"queryparts.dat")) as csvfile:
            nsclreader = csv.reader(csvfile, delimiter="|")
            for row in nsclreader:
                if row[0].startswith("#") or len(row) < 8:
                    continue
                nslc = {"network": row[0], "station": row[1],
                        "location": row[2], "channel": row[3],
                        "starttime": UTCDateTime(row[4]),
                        "endtime": UTCDateTime(row[5])}
                if row[6]:
                    returnval = row[6]
                else:
                    returnval = None
                    # could use row[7] to determine an appropriate error
                params_and_result.append({"nslc": nslc,"returnval": returnval})
        return params_and_result
            
                

    def test_known_quake(self):
        """
        Tests for queryauth
        These tests exist in this form because digest authentication may not
        confirm the authentication, even if credentials are valid, unless a
        specific server:port are accessed.

        """
        t1 = UTCDateTime("2010-02-27T06:30")
        t2 = UTCDateTime("2010-02-27T10:00")

        params_with_results = self.set_up_waveform_tests()
        for mytest in params_with_results:
            expected = mytest["returnval"]
            nslc = mytest["nslc"]
            if expected is not None:
                traces = self.client.get_waveform(**nslc)
                self.assertEqual(str(traces.count()), expected)
            else:
                try:
                    traces = self.client.get_waveform(**nslc)
                    print (traces)
                except Exception as e:
                    print e

        self.get_help_from_services()

        evparams = [
            {"starttime": t1, "endtime": t2, "minmagnitude": 1.0,
             "maxmagnitude": 9.0},
            {"starttime": t1, "endtime": t2, "minmagnitude": 8.0,
             "maxmagnitude": 7.0, "includeallmagnitudes": True},
            {"starttime": t1, "endtime": t2, "minmagnitude": 5.0,
             "includeallmagnitudes": True, "magnitudetype": "Mw"},
            {"starttime": t1, "endtime": t2, "includeallmagnitudes": "Yes",
             "maxmagnitude": 9.0},
            {"starttime": t1, "endtime": t2, "includeallmagnitudes": "Yes",
             "maxmagnitude": 9.0, "maxmag": 8.8}]
        svc_providers = ["IRIS", "IRIS", "USGS", "NCEDC", "rEsIf", "bob", None]
        to_debug = [True, True, False, False, False, False, True]
        fn = [None, "/Users/celsoreyes/Scripts/junk/cruft.out", None, None,
              None, None, None]
        staparams = [{"level": "network", "network": "A*,F*", "station": "",
                      "channel": "", "location": ""},
                     {"level": "network", "network": "A*,F*", "station": "",
                      "channel": "", "location": "  "},
                     {"level": "station", "network": "IU,FR", "station": "A*",
                      "channel": None, "location": ",00,--,,,"},
                     {"level": "channel", "network": "IU,FR",
                      "station": "ANMO", "channel": "?H?", "location": "?0"},
                     {"level": "station", "network": "*",
                      "station": "OKCF,AMNO,ANMO,AAA,TEST", "channel": "?H?",
                      "location": None},
                     {"level": "response", "network": "IU,FR",
                      "station": "ANMO", "channel": "*", "location": "?0"}]

        nslc = {"network": "IU", "station": "ANMO",
                "location": "00", "channel": "?HZ"}
        bulk = []
        bulk.append([("IU", "ANMO", None, "BHZ", t1, t2)])
        bulk.append([("IU", "ANMO", None, "BHZ", t1, t2),
                     ("AV", "O*", ",--,*", "SHZ,EHZ", t1, t2),
                     ("IU", "ANMO", "  ", "?H?", t1, t2)])
        bulk.append("""TA A25A -- BHZ 2010-03-25T00:00:00 2010-04-01T00:00:00
                    IU ANMO 00 BHZ 2010-03-25T00:00:00 2010-04-01T00:00:00
                    AF AAUS 00 BHZ 2009-05-03T00:00:00 2009-05-04T00:00:00
                    AF FOO 00 BHZ 2009-05-03T00:00:00 2009-05-04T00:00:00
                    7C XXX 00 BHO 1995-01-01T00:00:00 1995-01-02T00:00:00
                    """)

        bulk.append('quality=B\n' +
                    'longestonly=false\n' +
                    'IU ANMO * BHZ 2010-02-27 2010-02-27T00:00:02\n' +
                    'IU AFI 1? BHE 2010-02-27 2010-02-27T00:00:04\n' +
                    'GR GRA1 * BH? 2010-02-27 2010-02-27T00:00:02\n')

        for (datacenter, db) in zip(svc_providers, to_debug):
            try:
                self.client = Client(datacenter, debug=db)
            except Exception as e:
                print(e)
                continue
            try:
                print(self.client)
            except:
                pass

            self.get_help_from_services()
            for evparam in evparams:
                try:
                    ev = self.client.get_events(filename=fn, **evparam)
                    print(ev)
                except:
                    pass

            for staparam in staparams:
                try:
                    st = self.client.get_stations(filename=fn, **staparam)
                    print(st)
                except:
                    pass
            try:
                w = self.client.get_waveform(starttime=t1, endtime=t2,
                                             filename=fn, **nslc)
            except Exception as e:
                print(e)
                pass

            for b in bulk:
                try:
                    w = self.client.get_waveform_bulk(b, quality="B",
                                                      filename=fn)
                except Exception as e:
                    print(e)
                    pass
            try:
                self.client._build_url("base", 1, "badservice", "resource_ty")
            except Exception as e:
                print(e)
                pass
            try:
                self.client._build_url("base", 1, "badservice", "resource_ty",
                                       {"junk": nothing, "noerrors": True})
            except Exception as e:
                print(e)
                pass
            try:
                self.client._build_url("base", 1, "dataselect", "resource_ty",
                                       {"junk": nothing, "noerrors": True},
                                       branch_id="a1234bcd_f")
            except Exception as e:
                print(e)
                pass
            try:
                self.client._build_url("base", 1, "event", "resource_type",
                                       branch_id="#")
            except Exception as e:
                print(e)

            selectfile = os.path.join(self.datapath, "sampleselectionfile.txt")
            try:
                with open(selectfile) as f:
                    self.client.get_waveform_bulk(f)
            except Exception as e:
                print(e)
            try:
                self.client.get_waveform_bulk(selectfile)
            except Exception as e:
                print(e)


def suite():
    return unittest.makeSuite(QueryauthTestCase, 'test')

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
    #import doctest
    #doctest.testmod(exclude_empty=True)
