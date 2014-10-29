<?xml version='1.0' encoding='utf-8'?>
<q:quakeml xmlns:q="http://quakeml.org/xmlns/quakeml/1.2" xmlns="http://quakeml.org/xmlns/bed/1.2">
  <eventParameters publicID="smi:local/9f6e4efc-1dec-44a5-9c75-7092507eaf25">
    <creationInfo>
      <creationTime>2014-10-23T15:27:36.487786Z</creationTime>
      <version>ObsPy 0.9.2-1209-g0e9965dae0-dirty</version>
    </creationInfo>
    <event publicID="smi:local/cd1f535c-e75e-4dc6-8170-82e47cb40501">
      <creationInfo>
        <creationTime>2014-10-17T16:30:08.000000Z</creationTime>
        <version>NLLoc:v6.00.0</version>
      </creationInfo>
      <origin publicID="smi:local/3cb10e44-6aeb-4279-8caa-235452e5c9b3">
        <time>
          <value>2010-05-27T16:56:24.612600Z</value>
        </time>
        <latitude>
          <value>48.0470705175</value>
          <uncertainty>0.0012429</uncertainty>
        </latitude>
        <longitude>
          <value>11.6455375456</value>
          <uncertainty>0.0015118</uncertainty>
        </longitude>
        <depth>
          <value>4579.49</value>
          <uncertainty>191.606367326</uncertainty>
          <confidenceLevel>68</confidenceLevel>
        </depth>
        <depthType>from location</depthType>
        <quality>
          <associatedPhaseCount>8</associatedPhaseCount>
          <usedPhaseCount>8</usedPhaseCount>
          <associatedStationCount>-1</associatedStationCount>
          <usedStationCount>4</usedStationCount>
          <depthPhaseCount>-1</depthPhaseCount>
          <standardError>0.0189187</standardError>
          <azimuthalGap>129.214</azimuthalGap>
          <secondaryAzimuthalGap>129.214</secondaryAzimuthalGap>
          <groundTruthLevel>-</groundTruthLevel>
          <minimumDistance>0.0169035589727</minimumDistance>
          <maximumDistance>0.0751741131744</maximumDistance>
          <medianDistance>0.030874699985</medianDistance>
        </quality>
        <comment id="smi:local/93aa18a6-1ac9-4a65-b4cd-b8349ab1bc91">
            <text>Note: Depth/Latitude/Longitude errors are calculated from covariance matrix as 1D marginal (Lon/Lat errors as great circle degrees) while OriginUncertainty min/max horizontal errors are calculated from 2D error ellipsoid and are therefore seemingly higher compared to 1D errors. Error estimates can be reconstructed from the following original NonLinLoc error statistics line:
STATISTICS ExpectX 4473.68 Y 5323.29 Z 4.59501  CovXX 0.0282621 XY -0.0053866 XZ 0.0043871 YY 0.0191034 YZ 0.00503624 ZZ 0.036713 EllAz1  206.782 Dip1  16.4026 Len1  0.227982 Az2  300.149 Dip2  11.2855 Len2  0.327468 Len3  3.709256e-01</text>
        </comment>
        <creationInfo>
          <creationTime>2014-10-17T16:30:08.000000Z</creationTime>
          <version>NLLoc:v6.00.0</version>
        </creationInfo>
        <originUncertainty>
          <preferredDescription>uncertainty ellipse</preferredDescription>
          <minHorizontalUncertainty>195.472</minHorizontalUncertainty>
          <maxHorizontalUncertainty>265.954</maxHorizontalUncertainty>
          <azimuthMaxHorizontalUncertainty>114.816</azimuthMaxHorizontalUncertainty>
          <confidenceLevel>68.0</confidenceLevel>
        </originUncertainty>
        <arrival publicID="smi:local/677a937b-4db0-4fad-8822-9d75fdb3f9c5">
          <pickID>smi:local/d7ba3bb7-645f-4ee6-a8f4-65e0332c5025</pickID>
          <phase>P</phase>
          <azimuth>200.7</azimuth>
          <distance>0.0169036489048</distance>
          <takeoffAngle>
            <value>152.6</value>
          </takeoffAngle>
          <timeResidual>-0.015</timeResidual>
          <timeWeight>2.054</timeWeight>
        </arrival>
        <arrival publicID="smi:local/06e00087-a9c8-4ee5-a5a8-4240bcac7db8">
          <pickID>smi:local/3a0bde89-d7e6-45ef-a08a-4b950720f7be</pickID>
          <phase>S</phase>
          <azimuth>200.6</azimuth>
          <distance>0.0169036489048</distance>
          <takeoffAngle>
            <value>156.4</value>
          </takeoffAngle>
          <timeResidual>0.0195</timeResidual>
          <timeWeight>0.4108</timeWeight>
        </arrival>
        <arrival publicID="smi:local/619921cf-129e-4ef8-a743-a397becd7146">
          <pickID>smi:local/95492f79-0db4-4bba-a198-6c08db43dd83</pickID>
          <phase>P</phase>
          <azimuth>64.7</azimuth>
          <distance>0.0267620123489</distance>
          <takeoffAngle>
            <value>139.3</value>
          </takeoffAngle>
          <timeResidual>0.0076</timeResidual>
          <timeWeight>1.264</timeWeight>
        </arrival>
        <arrival publicID="smi:local/2c5ae0cc-981d-42c5-b2ef-57ae34ba5c02">
          <pickID>smi:local/75ca039c-bfca-41c3-8fd3-9d70341a23fe</pickID>
          <phase>S</phase>
          <azimuth>63.9</azimuth>
          <distance>0.0267620123489</distance>
          <takeoffAngle>
            <value>144.4</value>
          </takeoffAngle>
          <timeResidual>0.0156</timeResidual>
          <timeWeight>0.4108</timeWeight>
        </arrival>
        <arrival publicID="smi:local/44bbb2f9-76cc-4f73-9b8c-022ec9950b0b">
          <pickID>smi:local/550537fb-922c-4742-9bcc-504c1d330480</pickID>
          <phase>P</phase>
          <azimuth>348.8</azimuth>
          <distance>0.0349881070783</distance>
          <takeoffAngle>
            <value>131.0</value>
          </takeoffAngle>
          <timeResidual>-0.0085</timeResidual>
          <timeWeight>2.054</timeWeight>
        </arrival>
        <arrival publicID="smi:local/d7b02dea-3610-4209-9fe8-ab450a159c90">
          <pickID>smi:local/0783bf9f-6862-40a3-baeb-ed247b73ca6f</pickID>
          <phase>S</phase>
          <azimuth>348.4</azimuth>
          <distance>0.0349881070783</distance>
          <takeoffAngle>
            <value>137.7</value>
          </takeoffAngle>
          <timeResidual>0.0008</timeResidual>
          <timeWeight>1.264</timeWeight>
        </arrival>
        <arrival publicID="smi:local/f7d77c25-1729-4879-b014-2f4608b72844">
          <pickID>smi:local/d6fa4a8a-6e4c-48a1-9ab9-e84b60038f4b</pickID>
          <phase>P</phase>
          <azimuth>258.3</azimuth>
          <distance>0.0751742930387</distance>
          <takeoffAngle>
            <value>106.5</value>
          </takeoffAngle>
          <timeResidual>0.0663</timeResidual>
          <timeWeight>0.4108</timeWeight>
        </arrival>
        <arrival publicID="smi:local/dbeb77a9-1f2a-4d81-adb0-a1ad82713665">
          <pickID>smi:local/b0b74d7c-bd9e-4183-abcb-5c784a53f8b0</pickID>
          <phase>S</phase>
          <azimuth>258.1</azimuth>
          <distance>0.0751742930387</distance>
          <takeoffAngle>
            <value>113.1</value>
          </takeoffAngle>
          <timeResidual>-0.0299</timeResidual>
          <timeWeight>0.1315</timeWeight>
        </arrival>
      </origin>
      <pick publicID="smi:local/d7ba3bb7-645f-4ee6-a8f4-65e0332c5025">
        <time>
          <value>2010-05-27T16:56:25.930000Z</value>
          <uncertainty>0.02</uncertainty>
        </time>
        <waveformID stationCode="UH3"></waveformID>
        <phaseHint>P</phaseHint>
      </pick>
      <pick publicID="smi:local/3a0bde89-d7e6-45ef-a08a-4b950720f7be">
        <time>
          <value>2010-05-27T16:56:27.100000Z</value>
          <uncertainty>0.06</uncertainty>
        </time>
        <waveformID stationCode="UH3"></waveformID>
        <phaseHint>S</phaseHint>
      </pick>
      <pick publicID="smi:local/95492f79-0db4-4bba-a198-6c08db43dd83">
        <time>
          <value>2010-05-27T16:56:26.040000Z</value>
          <uncertainty>0.03</uncertainty>
        </time>
        <waveformID stationCode="UH2"></waveformID>
        <phaseHint>P</phaseHint>
      </pick>
      <pick publicID="smi:local/75ca039c-bfca-41c3-8fd3-9d70341a23fe">
        <time>
          <value>2010-05-27T16:56:27.270000Z</value>
          <uncertainty>0.06</uncertainty>
        </time>
        <waveformID stationCode="UH2"></waveformID>
        <phaseHint>S</phaseHint>
      </pick>
      <pick publicID="smi:local/550537fb-922c-4742-9bcc-504c1d330480">
        <time>
          <value>2010-05-27T16:56:26.130000Z</value>
          <uncertainty>0.02</uncertainty>
        </time>
        <waveformID stationCode="UH1"></waveformID>
        <phaseHint>P</phaseHint>
      </pick>
      <pick publicID="smi:local/0783bf9f-6862-40a3-baeb-ed247b73ca6f">
        <time>
          <value>2010-05-27T16:56:27.460000Z</value>
          <uncertainty>0.03</uncertainty>
        </time>
        <waveformID stationCode="UH1"></waveformID>
        <phaseHint>S</phaseHint>
      </pick>
      <pick publicID="smi:local/d6fa4a8a-6e4c-48a1-9ab9-e84b60038f4b">
        <time>
          <value>2010-05-27T16:56:26.930000Z</value>
          <uncertainty>0.06</uncertainty>
        </time>
        <waveformID stationCode="UH4"></waveformID>
        <phaseHint>P</phaseHint>
      </pick>
      <pick publicID="smi:local/b0b74d7c-bd9e-4183-abcb-5c784a53f8b0">
        <time>
          <value>2010-05-27T16:56:28.900000Z</value>
          <uncertainty>0.11</uncertainty>
        </time>
        <waveformID stationCode="UH4"></waveformID>
        <phaseHint>S</phaseHint>
      </pick>
    </event>
  </eventParameters>
</q:quakeml>
