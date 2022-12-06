<?xml version='1.0' encoding='utf-8'?>
<q:quakeml xmlns:q="http://quakeml.org/xmlns/quakeml/1.2" xmlns="http://quakeml.org/xmlns/bed/1.2">
  <eventParameters publicID="smi:local/f0882fec-3cdb-4f9a-9729-b41627fac8eb">
    <creationInfo>
      <creationTime>2016-05-03T20:00:55.781589Z</creationTime>
      <version>ObsPy 1.0.1.post0+20.gbeb9f46723.dirty</version>
    </creationInfo>
    <event publicID="smi:local/0eee2e6f-064b-458a-934f-c5d3105e9529">
      <preferredOriginID>smi:local/a2260002-95c6-42f7-8c44-f46124355228</preferredOriginID>
      <creationInfo>
        <creationTime>2013-06-21T12:53:34.000000Z</creationTime>
        <version>NLLoc:v6.02.07</version>
      </creationInfo>
      <origin publicID="smi:local/a2260002-95c6-42f7-8c44-f46124355228">
        <time>
          <value>2006-07-15T17:21:20.195670Z</value>
        </time>
        <latitude>
          <value>51.657659</value>
          <uncertainty>0.00724156630677</uncertainty>
        </latitude>
        <longitude>
          <value>7.736781</value>
          <uncertainty>0.00989286468574</uncertainty>
        </longitude>
        <depth>
          <value>1433.59</value>
          <uncertainty>1153.92807402</uncertainty>
          <confidenceLevel>68</confidenceLevel>
        </depth>
        <depthType>from location</depthType>
        <quality>
          <associatedPhaseCount>5</associatedPhaseCount>
          <usedPhaseCount>5</usedPhaseCount>
          <associatedStationCount>-1</associatedStationCount>
          <usedStationCount>5</usedStationCount>
          <depthPhaseCount>-1</depthPhaseCount>
          <standardError>0.00394121</standardError>
          <azimuthalGap>156.347</azimuthalGap>
          <secondaryAzimuthalGap>227.423</secondaryAzimuthalGap>
          <groundTruthLevel>-</groundTruthLevel>
          <minimumDistance>0.00329945808744</minimumDistance>
          <maximumDistance>0.0145801616038</maximumDistance>
          <medianDistance>0.00802668814966</medianDistance>
        </quality>
        <comment id="smi:local/3aadb009-ca54-4843-9fc1-57f1561295a3">
          <text>Note: Depth/Latitude/Longitude errors are calculated from covariance matrix as 1D marginal (Lon/Lat errors as great circle degrees) while OriginUncertainty min/max horizontal errors are calculated from 2D error ellipsoid and are therefore seemingly higher compared to 1D errors. Error estimates can be reconstructed from the following original NonLinLoc error statistics line:
STATISTICS ExpectX -1.32658 Y -0.0487098 Z 3.12781  CovXX 1.21008 XY 0.238028 XZ -0.486034 YY 0.648388 YZ -0.0503814 ZZ 1.33155 EllAz1  331.493 Dip1  -13.2202 Len1  1.37113 Az2  229.814 Dip2  -40.7512 Len2  1.74531 Len3  2.516878e+00</text>
        </comment>
        <creationInfo>
          <creationTime>2013-06-21T12:53:34.000000Z</creationTime>
          <version>NLLoc:v6.02.07</version>
        </creationInfo>
        <originUncertainty>
          <preferredDescription>uncertainty ellipse</preferredDescription>
          <minHorizontalUncertainty>1136.0</minHorizontalUncertainty>
          <maxHorizontalUncertainty>1727.42</maxHorizontalUncertainty>
          <azimuthMaxHorizontalUncertainty>69.8588</azimuthMaxHorizontalUncertainty>
          <confidenceLevel>68.0</confidenceLevel>
        </originUncertainty>
        <arrival publicID="smi:local/4822394a-8dba-4b4d-b8d6-47c75d94a600">
          <pickID>smi:local/80f620bf-5550-4fc5-b1a6-5d4394795878</pickID>
          <phase>P</phase>
          <azimuth>109.48</azimuth>
          <distance>0.00329961097212</distance>
          <timeResidual>-0.0076</timeResidual>
          <timeWeight>0.9958</timeWeight>
        </arrival>
        <arrival publicID="smi:local/0217b7ba-a9f7-46c8-b9c9-1c3de497a965">
          <pickID>smi:local/804b43a8-fe67-4041-af14-be0a2ea3e493</pickID>
          <phase>P</phase>
          <azimuth>13.71</azimuth>
          <distance>0.00340932820804</distance>
          <timeResidual>0.0025</timeResidual>
          <timeWeight>1.0009</timeWeight>
        </arrival>
        <arrival publicID="smi:local/c0778a50-dd6e-4080-9761-2aae8bbbcdab">
          <pickID>smi:local/9254790e-4e24-415f-b7a2-a60e504f3549</pickID>
          <phase>P</phase>
          <azimuth>71.75</azimuth>
          <distance>0.00396331031728</distance>
          <timeResidual>-0.0009</timeResidual>
          <timeWeight>1.0016</timeWeight>
        </arrival>
        <arrival publicID="smi:local/963aa4e6-2b6e-4978-8fcc-de9a9ec4891a">
          <pickID>smi:local/f744786a-bc96-4476-8c5c-6f2a9a5fef54</pickID>
          <phase>P</phase>
          <azimuth>204.61</azimuth>
          <distance>0.00499483219927</distance>
          <timeResidual>0.0065</timeResidual>
          <timeWeight>0.997</timeWeight>
        </arrival>
        <arrival publicID="smi:local/898783bb-82eb-45e6-84d1-86ca16c536bd">
          <pickID>smi:local/a040ecfb-0d12-4c91-9e37-463f075a2ec6</pickID>
          <phase>P</phase>
          <azimuth>104.59</azimuth>
          <distance>0.00563874646911</distance>
          <timeResidual>-0.0005</timeResidual>
          <timeWeight>1.0016</timeWeight>
        </arrival>
      </origin>
      <pick publicID="smi:local/80f620bf-5550-4fc5-b1a6-5d4394795878">
        <time>
          <value>2006-07-15T17:21:20.630000Z</value>
          <uncertainty>0.05</uncertainty>
        </time>
        <waveformID stationCode="HM02"></waveformID>
        <onset>impulsive</onset>
        <phaseHint>P</phaseHint>
        <polarity>positive</polarity>
      </pick>
      <pick publicID="smi:local/804b43a8-fe67-4041-af14-be0a2ea3e493">
        <time>
          <value>2006-07-15T17:21:20.640000Z</value>
          <uncertainty>0.05</uncertainty>
        </time>
        <waveformID stationCode="HM04"></waveformID>
        <onset>impulsive</onset>
        <phaseHint>P</phaseHint>
        <polarity>positive</polarity>
      </pick>
      <pick publicID="smi:local/9254790e-4e24-415f-b7a2-a60e504f3549">
        <time>
          <value>2006-07-15T17:21:20.640000Z</value>
          <uncertainty>0.05</uncertainty>
        </time>
        <waveformID stationCode="HM05"></waveformID>
        <onset>impulsive</onset>
        <phaseHint>P</phaseHint>
        <polarity>positive</polarity>
      </pick>
      <pick publicID="smi:local/f744786a-bc96-4476-8c5c-6f2a9a5fef54">
        <time>
          <value>2006-07-15T17:21:20.660000Z</value>
          <uncertainty>0.05</uncertainty>
        </time>
        <waveformID stationCode="HM10"></waveformID>
        <onset>impulsive</onset>
        <phaseHint>P</phaseHint>
        <polarity>positive</polarity>
      </pick>
      <pick publicID="smi:local/a040ecfb-0d12-4c91-9e37-463f075a2ec6">
        <time>
          <value>2006-07-15T17:21:20.660000Z</value>
          <uncertainty>0.05</uncertainty>
        </time>
        <waveformID stationCode="HM08"></waveformID>
        <onset>impulsive</onset>
        <phaseHint>P</phaseHint>
        <polarity>positive</polarity>
      </pick>
    </event>
  </eventParameters>
</q:quakeml>
