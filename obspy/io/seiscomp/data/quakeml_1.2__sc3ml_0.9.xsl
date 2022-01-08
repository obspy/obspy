<?xml version="1.0" encoding="UTF-8"?>
<!--
    ***************************************************************************

QuakeML 1.2 to SC3ML 0.9 stylesheet converter

Author:
    EOST (École et Observatoire des Sciences de la Terre)
Copyright:
    The ObsPy Development Team (devs@obspy.org)
License:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)

Usage
=====

This stylesheet converts a QuakeML to a SC3ML document. It may be invoked using
xalan or xsltproc:

    xalan -in quakeml.xml -xsl quakeml_1.2__sc3ml_0.9.xsl -out sc3ml.xml
    xsltproc quakeml_1.2__sc3ml_0.9.xsl quakeml.xml -o sc3ml.xml

Transformation
==============

QuakeML and SC3ML are quite similar schemas. Nevertheless some differences
exist.

ID restrictions
```````````````

SC3ML does not enforce any particular ID restriction unlike QuakeML. It isn't
a problem to convert from QuakeML to SC3ML.

Repositioning and creation of nodes
```````````````````````````````````

In SC3ML all information is grouped under the EventParameters element.

    <EventParameters>               <eventParameters>
                                        <event>
        <pick/>                             <pick/>
        <amplitude/>                        <amplitude/>
        <reading/>
        <origin>                            <origin/>
            <stationMagnitude/>             <stationMagnitude/>
            <magnitude/>                    <magnitude/>
        </origin>
        <focalMechanism/>                   <focalMechanism/>
        <event/>                        </event>
    </EventParameters>              </eventParameters

Since origins and focalMechanism aren't in an event anymore, OriginReferences
and FocalMechanismReferences need to be created.

Some nodes are also mandatory in SC3ML which aren't in QuakeML:
- event/description/type
- dataUsed/stationCount
- dataUsed/componentCount
- amplitude/type

Renaming of nodes
`````````````````

The following table lists the mapping of names between both schema:

Parent              QuakeML name                SC3 name
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
seiscomp            eventParameters             EventParameters
arrival             timeWeight                  weight
arrival             takeoffAngle (RealQuantity) takeOffAngle (double)
magnitude           mag                         magnitude
stationMagnitude    mag                         magnitude
amplitude           genericAmplitude            amplitude
origin              originUncertainty           uncertainty
momentTensor        category                    method
comment             @id (attribute)             id
waveformID          text()                      resourceURI

Enumerations
````````````

Both schema use enumerations. Numerous mappings are applied.

Unit conversion
```````````````

QuakeML uses meter for origin depth, origin uncertainty and confidence
ellipsoid, SC3ML uses kilometer.

Unmapped node
`````````````

The following nodes can not be mapped to the SC3ML schema, thus their data is
lost:

Parent              Element lost
''''''''''''''''''''''''''''''''''''''''''''
amplitude           evaluationStatus
magnitude           evaluationMode
originUncertainty   confidenceLevel
arrival             commment
arrival             horizontalSlownessWeight
arrival             backazimuthWeight
origin              region
dataUsed            longestPeriod
momentTensor        inversionType
focalMechanism      waveformID

Nodes order
```````````

Unlike SC3ML, QuakeML nodes can appear in any order. They must be reordered for
SC3ML. Unnecessary attributes must also be removed.

    ***************************************************************************
-->
<xsl:stylesheet version="1.0"
        xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
        xmlns:xs="http://www.w3.org/2001/XMLSchema"
        xmlns:ext="http://exslt.org/common"
        xmlns="http://geofon.gfz-potsdam.de/ns/seiscomp3-schema/0.9"
        xmlns:qml="http://quakeml.org/xmlns/bed/1.2"
        xmlns:q="http://quakeml.org/xmlns/quakeml/1.2"
        exclude-result-prefixes="xsl xs ext q qml">
    <xsl:output method="xml" encoding="UTF-8" indent="yes"/>
    <xsl:strip-space elements="*"/>

    <!-- Define some global variables -->
    <xsl:variable name="version" select="0.9"/>
    <xsl:variable name="schema" select="document('sc3ml_0.9.xsd')"/>

    <!-- Define key to remove duplicates-->
    <xsl:key name="pick_key" match="qml:pick" use="@publicID"/>
    <xsl:key name="amplitude_key" match="qml:amplitude" use="@publicID"/>
    <xsl:key name="origin_key" match="qml:origin" use="@publicID"/>
    <xsl:key name="focalMechanism_key" match="qml:focalMechanism" use="@publicID"/>

<!--
    ***************************************************************************
    Move/create nodes
    ***************************************************************************
-->

    <!-- Default match: Map node 1:1 -->
    <xsl:template match="*">
        <xsl:element name="{local-name()}">
            <xsl:copy-of select="@*"/>
            <xsl:apply-templates select="node()"/>
        </xsl:element>
    </xsl:template>

    <!-- Starting point: Match the root node and select the EventParameters
         node -->
    <xsl:template match="/">
        <!-- Write a disordered sc3ml in this variable. It will be ordered in
             a second time. -->
        <xsl:variable name="disordered">
            <xsl:for-each select="./q:quakeml/qml:eventParameters">
                <EventParameters>
                    <xsl:copy-of select="@publicID"/>
                    <xsl:apply-templates/>
                </EventParameters>
            </xsl:for-each>
        </xsl:variable>

        <!-- Reorder nodes -->
        <seiscomp version="{$version}">
            <xsl:apply-templates select="ext:node-set($disordered)" mode="reorder"/>
        </seiscomp>
    </xsl:template>

    <xsl:template match="qml:event">
        <!-- Create event node -->
        <xsl:element name="{local-name()}">
            <xsl:copy-of select="@*"/>
            <xsl:apply-templates/>

            <!-- Create origin references -->
            <xsl:for-each select="qml:origin[count(. | key('origin_key', @publicID)[1]) = 1]">
                <xsl:element name="originReference">
                    <xsl:value-of select="@publicID"/>
                </xsl:element>
            </xsl:for-each>

            <!-- Create focal mechanism references -->
            <xsl:for-each select="qml:focalMechanism[count(. | key('focalMechanism_key', @publicID)[1]) = 1]">
                <xsl:element name="focalMechanismReference">
                    <xsl:value-of select="@publicID"/>
                </xsl:element>
            </xsl:for-each>
        </xsl:element>

        <!-- Copy picks and remove duplicates -->
        <xsl:for-each select="qml:pick[count(. | key('pick_key', @publicID)[1]) = 1]">
            <xsl:element name="{local-name()}">
                <xsl:copy-of select="@*"/>
                <xsl:apply-templates/>
            </xsl:element>
        </xsl:for-each>

        <!-- Copy amplitudes and remove duplicates -->
        <xsl:for-each select="qml:amplitude[count(. | key('amplitude_key', @publicID)[1]) = 1]">
            <xsl:element name="{local-name()}">
                <xsl:copy-of select="@*"/>
                <xsl:if test="not(qml:type)">
                    <xsl:element name="type"/>
                </xsl:if>
                <xsl:apply-templates/>
            </xsl:element>
        </xsl:for-each>

        <!-- Copy origins and remove duplicates -->
        <xsl:for-each select="qml:origin[count(. | key('origin_key', @publicID)[1]) = 1]">
            <xsl:element name="{local-name()}">
                <xsl:copy-of select="@*"/>

                <!-- Copy magnitudes and remove duplicates -->
                <xsl:for-each select="../qml:magnitude[qml:originID/text()=current()/@publicID]">
                    <xsl:element name="{local-name()}">
                        <xsl:copy-of select="@*"/>
                        <xsl:apply-templates/>
                    </xsl:element>
                </xsl:for-each>

                <!-- Copy stations magnitudes and remove duplicates -->
                <xsl:for-each select="../qml:stationMagnitude[qml:originID/text()=current()/@publicID]">
                    <xsl:element name="{local-name()}">
                        <xsl:copy-of select="@*"/>
                        <xsl:apply-templates/>
                    </xsl:element>
                </xsl:for-each>

                <xsl:apply-templates/>
            </xsl:element>
        </xsl:for-each>

        <!-- Copy focal mechanisms and remove duplicates -->
        <xsl:for-each select="qml:focalMechanism[count(. | key('focalMechanism_key', @publicID)[1]) = 1]">
            <xsl:element name="{local-name()}">
                <xsl:copy-of select="@*"/>
                <xsl:apply-templates/>
            </xsl:element>
        </xsl:for-each>
    </xsl:template>

    <!-- Create mandatory element event/description/type -->
    <xsl:template match="qml:event/qml:description">
        <xsl:element name="{local-name()}">
            <xsl:if test="not(qml:type)">
                <xsl:element name="type">region name</xsl:element>
            </xsl:if>
            <xsl:apply-templates/>
        </xsl:element>
    </xsl:template>

    <!-- Create mandatory elements dataUsed/stationCount and
         dataUsed/componentCount -->
    <xsl:template match="qml:dataUsed">
        <xsl:element name="{local-name()}">
            <xsl:if test="not(qml:stationCount)">
                <xsl:element name="stationCount"/>
            </xsl:if>
            <xsl:if test="not(qml:componentCount)">
                <xsl:element name="componentCount"/>
            </xsl:if>
            <xsl:apply-templates/>
        </xsl:element>
    </xsl:template>

<!--
    ***************************************************************************
    Rename nodes/attributes
    ***************************************************************************
-->

    <!-- arrival/timeWeight -> arrival/weight -->
    <xsl:template match="qml:arrival/qml:timeWeight">
        <xsl:element name="weight">
            <xsl:apply-templates/>
        </xsl:element>
    </xsl:template>

    <!-- arrival/takeoffAngle -> arrival/takeOffAngle -->
    <xsl:template match="qml:arrival/qml:takeoffAngle">
        <xsl:element name="takeOffAngle">
            <xsl:value-of select="qml:value"/>
        </xsl:element>
    </xsl:template>

    <!-- magnitude/mag -> magnitude/magnitude -->
    <!-- stationMagnitude/mag -> stationMagnitude/magnitude -->
    <xsl:template match="qml:magnitude/qml:mag
                         | qml:stationMagnitude/qml:mag">
        <xsl:element name="magnitude">
            <xsl:apply-templates/>
        </xsl:element>
    </xsl:template>

    <!-- amplitude/genericAmplitutde -> amplitude/amplitude -->
    <xsl:template match="qml:amplitude/qml:genericAmplitude">
        <xsl:element name="amplitude">
            <xsl:copy-of select="@*"/>
            <xsl:apply-templates/>
        </xsl:element>
    </xsl:template>

    <!-- origin/originUncertainty -> origin/uncertainty -->
    <xsl:template match="qml:origin/qml:originUncertainty">
        <xsl:element name="uncertainty">
            <xsl:copy-of select="@*"/>
            <xsl:apply-templates/>
        </xsl:element>
    </xsl:template>

    <!-- momentTensor/category -> momentTensor/method -->
    <xsl:template match="qml:momentTensor/qml:category">
        <xsl:element name="method">
            <xsl:apply-templates/>
        </xsl:element>
    </xsl:template>

    <!-- comment/@id -> comment/id -->
    <xsl:template match="qml:comment">
        <xsl:element name="{local-name()}">
            <xsl:if test="@id != ''">
                <xsl:element name="id">
                    <xsl:value-of select="@id"/>
                </xsl:element>
            </xsl:if>
            <xsl:apply-templates/>
        </xsl:element>
    </xsl:template>

    <!-- waveformID/text() -> waveformID/resourceURI -->
    <xsl:template match="qml:waveformID">
        <xsl:element name="{local-name()}">
            <xsl:copy-of select="@*"/>
            <xsl:if test="current() != ''">
                <xsl:element name="resourceURI">
                    <xsl:value-of select="."/>
                </xsl:element>
            </xsl:if>
        </xsl:element>
    </xsl:template>

<!--
    ***************************************************************************
    Enumeration mapping
    ***************************************************************************
-->

    <xsl:template match="qml:event/qml:type">
        <xsl:element name="{local-name()}">
            <xsl:variable name="v" select="."/>
            <xsl:choose>
                <xsl:when test="$v='not reported'">other</xsl:when>
                <xsl:when test="$v='anthropogenic event'">other</xsl:when>
                <xsl:when test="$v='collapse'">other</xsl:when>
                <xsl:when test="$v='cavity collapse'">other</xsl:when>
                <xsl:when test="$v='accidental explosion'">other</xsl:when>
                <xsl:when test="$v='controlled explosion'">other</xsl:when>
                <xsl:when test="$v='experimental explosion'">other</xsl:when>
                <xsl:when test="$v='industrial explosion'">other</xsl:when>
                <xsl:when test="$v='mining explosion'">other</xsl:when>
                <xsl:when test="$v='road cut'">other</xsl:when>
                <xsl:when test="$v='blasting levee'">other</xsl:when>
                <xsl:when test="$v='induced or triggered event'">induced earthquake</xsl:when>
                <xsl:when test="$v='rock burst'">other</xsl:when>
                <xsl:when test="$v='reservoir loading'">other</xsl:when>
                <xsl:when test="$v='fluid injection'">other</xsl:when>
                <xsl:when test="$v='fluid extraction'">other</xsl:when>
                <xsl:when test="$v='crash'">other</xsl:when>
                <xsl:when test="$v='train crash'">other</xsl:when>
                <xsl:when test="$v='boat crash'">other</xsl:when>
                <xsl:when test="$v='other event'">other</xsl:when>
                <xsl:when test="$v='atmospheric event'">other</xsl:when>
                <xsl:when test="$v='sonic blast'">other</xsl:when>
                <xsl:when test="$v='acoustic noise'">other</xsl:when>
                <xsl:when test="$v='thunder'">other</xsl:when>
                <xsl:when test="$v='avalanche'">other</xsl:when>
                <xsl:when test="$v='hydroacoustic event'">other</xsl:when>
                <xsl:when test="$v='ice quake'">other</xsl:when>
                <xsl:when test="$v='slide'">other</xsl:when>
                <xsl:when test="$v='meteorite'">meteor impact</xsl:when>
                <xsl:otherwise><xsl:value-of select="$v"/></xsl:otherwise>
            </xsl:choose>
        </xsl:element>
    </xsl:template>

    <xsl:template match="qml:origin/qml:depthType">
        <xsl:element name="{local-name()}">
            <xsl:variable name="v" select="."/>
            <xsl:choose>
                <xsl:when test="$v='constrained by depth and direct phases'">other</xsl:when>
                <xsl:otherwise><xsl:value-of select="$v"/></xsl:otherwise>
            </xsl:choose>
        </xsl:element>
    </xsl:template>

    <xsl:template match="qml:momentTensor/qml:dataUsed/qml:waveType">
        <xsl:element name="{local-name()}">
            <xsl:variable name="v" select="."/>
            <xsl:choose>
                <xsl:when test="$v='P waves'">P body waves</xsl:when>
                <xsl:when test="$v='mantle waves'">long-period mantle waves</xsl:when>
                <xsl:when test="$v='combined'">unknown</xsl:when>
                <xsl:otherwise><xsl:value-of select="$v"/></xsl:otherwise>
            </xsl:choose>
        </xsl:element>
    </xsl:template>

<!--
    ***************************************************************************
    Unit conversion
    ***************************************************************************
-->

    <!-- Origin depth, SC3ML uses kilometer, QuakeML meter -->
    <xsl:template match="qml:origin/qml:depth/qml:value
                         | qml:origin/qml:depth/qml:uncertainty
                         | qml:origin/qml:depth/qml:lowerUncertainty
                         | qml:origin/qml:depth/qml:upperUncertainty
                         | qml:origin/qml:originUncertainty/qml:horizontalUncertainty
                         | qml:origin/qml:originUncertainty/qml:minHorizontalUncertainty
                         | qml:origin/qml:originUncertainty/qml:maxHorizontalUncertainty
                         | qml:confidenceEllipsoid/qml:semiMajorAxisLength
                         | qml:confidenceEllipsoid/qml:semiMinorAxisLength
                         | qml:confidenceEllipsoid/qml:semiIntermediateAxisLength">
        <xsl:element name="{local-name()}">
            <xsl:value-of select="current() div 1000"/>
        </xsl:element>
    </xsl:template>

<!--
    ***************************************************************************
    Delete moved/unmapped nodes
    ***************************************************************************
-->

    <xsl:template match="qml:pick"/>
    <xsl:template match="qml:amplitude"/>
    <xsl:template match="qml:origin"/>
    <xsl:template match="qml:magnitude"/>
    <xsl:template match="qml:stationMagnitude"/>
    <xsl:template match="qml:focalMechanism"/>
    <xsl:template match="qml:amplitude/qml:category"/>
    <xsl:template match="qml:amplitude/qml:evaluationStatus"/>
    <xsl:template match="qml:magnitude/qml:evaluationMode"/>
    <xsl:template match="qml:originUncertainty/qml:confidenceLevel"/>
    <xsl:template match="qml:arrival/qml:comment"/>
    <xsl:template match="qml:arrival/qml:horizontalSlownessWeight"/>
    <xsl:template match="qml:arrival/qml:backazimuthWeight"/>
    <xsl:template match="qml:origin/qml:region"/>
    <xsl:template match="qml:dataUsed/qml:longestPeriod"/>
    <xsl:template match="qml:momentTensor/qml:inversionType"/>
    <xsl:template match="qml:focalMechanism/qml:waveformID"/>

<!--
    ***************************************************************************
    Reorder element nodes
    ***************************************************************************
-->

    <xsl:template match="*" mode="reorder">
        <!-- Detect complexType from node name -->
        <xsl:variable name="name">
            <xsl:variable name="v" select="local-name()"/>
            <xsl:variable name="p" select="local-name(..)"/>
            <xsl:choose>
                <xsl:when test="$v='scalingTime'">TimeQuantity</xsl:when>
                <xsl:when test="$v='time'">TimeQuantity</xsl:when>
                <xsl:when test="$v='creationInfo'">CreationInfo</xsl:when>
                <xsl:when test="$p='event' and $v='description'">EventDescription</xsl:when>
                <xsl:when test="$v='comment'">Comment</xsl:when>
                <xsl:when test="$p='tAxis' and $v='azimuth'">RealQuantity</xsl:when>
                <xsl:when test="$p='pAxis' and $v='azimuth'">RealQuantity</xsl:when>
                <xsl:when test="$p='nAxis' and $v='azimuth'">RealQuantity</xsl:when>
                <xsl:when test="$v='plunge'">RealQuantity</xsl:when>
                <xsl:when test="$v='length'">RealQuantity</xsl:when>
                <xsl:when test="$v='second'">RealQuantity</xsl:when>
                <xsl:when test="$v='Mrr'">RealQuantity</xsl:when>
                <xsl:when test="$v='Mtt'">RealQuantity</xsl:when>
                <xsl:when test="$v='Mpp'">RealQuantity</xsl:when>
                <xsl:when test="$v='Mrt'">RealQuantity</xsl:when>
                <xsl:when test="$v='Mrp'">RealQuantity</xsl:when>
                <xsl:when test="$v='Mtp'">RealQuantity</xsl:when>
                <xsl:when test="$v='strike'">RealQuantity</xsl:when>
                <xsl:when test="$p='nodalPlane1' and $v='dip'">RealQuantity</xsl:when>
                <xsl:when test="$p='nodalPlane2' and $v='dip'">RealQuantity</xsl:when>
                <xsl:when test="$v='rake'">RealQuantity</xsl:when>
                <xsl:when test="$v='scalarMoment'">RealQuantity</xsl:when>
                <xsl:when test="$p='amplitude' and $v='amplitude'">RealQuantity</xsl:when>
                <xsl:when test="$v='period'">RealQuantity</xsl:when>
                <xsl:when test="$p='magnitude' and $v='magnitude'">RealQuantity</xsl:when>
                <xsl:when test="$p='stationMagnitude' and $v='magnitude'">RealQuantity</xsl:when>
                <xsl:when test="$v='horizontalSlowness'">RealQuantity</xsl:when>
                <xsl:when test="$v='backazimuth'">RealQuantity</xsl:when>
                <xsl:when test="$p='origin' and $v='latitude'">RealQuantity</xsl:when>
                <xsl:when test="$p='origin' and $v='longitude'">RealQuantity</xsl:when>
                <xsl:when test="$p='origin' and $v='depth'">RealQuantity</xsl:when>
                <xsl:when test="$v='year'">IntegerQuantity</xsl:when>
                <xsl:when test="$v='month'">IntegerQuantity</xsl:when>
                <xsl:when test="$v='day'">IntegerQuantity</xsl:when>
                <xsl:when test="$v='hour'">IntegerQuantity</xsl:when>
                <xsl:when test="$v='minute'">IntegerQuantity</xsl:when>
                <xsl:when test="$v='tAxis'">Axis</xsl:when>
                <xsl:when test="$v='pAxis'">Axis</xsl:when>
                <xsl:when test="$v='nAxis'">Axis</xsl:when>
                <xsl:when test="$v='principalAxes'">PrincipalAxes</xsl:when>
                <xsl:when test="$v='dataUsed'">DataUsed</xsl:when>
                <xsl:when test="$v='compositeTime'">CompositeTime</xsl:when>
                <xsl:when test="$v='tensor'">Tensor</xsl:when>
                <xsl:when test="$v='quality'">OriginQuality</xsl:when>
                <xsl:when test="$v='nodalPlane1'">NodalPlane</xsl:when>
                <xsl:when test="$v='nodalPlane2'">NodalPlane</xsl:when>
                <xsl:when test="$v='timeWindow'">TimeWindow</xsl:when>
                <xsl:when test="$v='waveformID'">WaveformStreamID</xsl:when>
                <xsl:when test="$v='sourceTimeFunction'">SourceTimeFunction</xsl:when>
                <xsl:when test="$v='nodalPlanes'">NodalPlanes</xsl:when>
                <xsl:when test="$v='confidenceEllipsoid'">ConfidenceEllipsoid</xsl:when>
                <xsl:when test="$v='reading'">Reading</xsl:when>
                <xsl:when test="$v='component'">MomentTensorComponentContribution</xsl:when>
                <xsl:when test="$v='stationMomentTensorContribution'">MomentTensorStationContribution</xsl:when>
                <xsl:when test="$v='phaseSetting'">MomentTensorPhaseSetting</xsl:when>
                <xsl:when test="$v='momentTensor'">MomentTensor</xsl:when>
                <xsl:when test="$v='focalMechanism'">FocalMechanism</xsl:when>
                <xsl:when test="$p='EventParameters' and $v='amplitude'">Amplitude</xsl:when>
                <xsl:when test="$v='stationMagnitudeContribution'">StationMagnitudeContribution</xsl:when>
                <xsl:when test="$p='origin' and $v='magnitude'">Magnitude</xsl:when>
                <xsl:when test="$v='stationMagnitude'">StationMagnitude</xsl:when>
                <xsl:when test="$v='pick'">Pick</xsl:when>
                <xsl:when test="$v='event'">Event</xsl:when>
                <xsl:when test="$p='origin' and $v='uncertainty'">OriginUncertainty</xsl:when>
                <xsl:when test="$v='arrival'">Arrival</xsl:when>
                <xsl:when test="$v='origin'">Origin</xsl:when>
                <xsl:when test="$v='EventParameters'">EventParameters</xsl:when>
                <xsl:otherwise/>
            </xsl:choose>
        </xsl:variable>
        <xsl:variable name="current" select="."/>

        <xsl:element name="{local-name()}">
            <xsl:choose>
                <xsl:when test="$name=''">
                    <!-- Not a complexType, don't reorder -->
                    <xsl:apply-templates select="node()" mode="reorder"/>
                </xsl:when>
                <xsl:otherwise>
                    <!-- This node is a complexType. -->
                    <!-- Only copy allowed attributes -->
                    <xsl:for-each select="$schema//xs:complexType[@name=$name]
                                          /xs:attribute/@name">
                        <xsl:copy-of select="$current/@*[local-name()=current()]"/>
                    </xsl:for-each>
                    <!-- Reorder nodes according to the XSD -->
                    <xsl:for-each select="$schema//xs:complexType[@name=$name]
                                          /xs:sequence/xs:element/@name">
                        <xsl:apply-templates select="$current/*[local-name()=current()]"
                                             mode="reorder"/>
                    </xsl:for-each>
                </xsl:otherwise>
            </xsl:choose>
        </xsl:element>
    </xsl:template>
</xsl:stylesheet>
