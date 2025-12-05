<?xml version="1.0" encoding="UTF-8"?>
<!--
QuakeML 1.2 to SCML (SeisComPML) 0.8 stylesheet converter

Author:
    EOST (École et Observatoire des Sciences de la Terre)
    Stephan Herrnkind <herrnkind@gempa.de>
Copyright:
    The ObsPy Development Team (devs@obspy.org)
License:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)

Usage
=====

This stylesheet converts a QuakeML to a SCML document. It may be invoked using
xalan or xsltproc:

    xalan -in quakeml.xml -xsl quakeml_1.2__sc3ml_0.8.xsl -out sc3ml.xml
    xsltproc quakeml_1.2__sc3ml_0.8.xsl quakeml.xml > sc3ml.xml

Due to the QuakeML ID schema the public IDs used by QuakeML are rather long
and may cause problems in SeisComP applications when displaying or processing
them. Especially the slash causes problems, e.g., when an event ID is used on
the command line or in a directory structure. To remove the ID prefix during
the conversion you may use the ID_PREFIX parameter:

    xalan -param ID_PREFIX "smi:org.gfz-potsdam.de/geofon/" -in quakeml.xml -xsl quakeml_1.2__sc3ml_0.8.xsl -out scml.xml
    xsltproc -stringparam ID_PREFIX smi:org.gfz-potsdam.de/geofon/ quakeml_1.2__sc3ml_0.8.xsl quakeml.xml > scml.xml

Other variable exist which control
  - the eventID format (BUILD_EVENT_ID),
  - the mapping of Magnitudes to Origins (GUESS_MAG_ORIGIN) or
  - the creation of unique ids (EVENT_INFO_ID)


Profiles
````````

The following table collects recommendated parameters setting for the QuakeML
import from different agencies:

Agency   ID_PREFIX             BUILD_EVENT_ID  GUESS_MAG_ORIGIN  EVENT_INFO_ID
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
USGS     quakeml:us.anss.org/  2               3                 1


Transformation
==============

QuakeML and SCML are quite similar schemas. Nevertheless some differences
exist.


ID restrictions
```````````````

SCML does not enforce any particular ID restriction unlike QuakeML. It isn't
a problem to convert from QuakeML to SCML. However, the QuakeML ID prefix may
be removed during the conversion, see ID_PREFIX variable.


Repositioning and creation of elements
``````````````````````````````````````

QuakeML groups all elements under the event element where SCML places elements
which might exist independent of an event, such as picks, amplitudes or origins
directly under the EventParameters element. Furthermore SCML does not support
magnitudes independent of an origin.

QuakeML                        SCML
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
<eventParameters>              <EventParameters>
    <event>
        <pick/>                    <pick/>
        <amplitude/>               <amplitude/>
                                   <reading/>
        <origin/>                  <origin>
        <stationMagnitude/>            <stationMagnitude/>
        <magnitude/>                   <magnitude/>
                                    </origin>
        <focalMechanism/>          <focalMechanism/>
    </event>                       <event/>
</eventParameters>             </EventParameters>

In SCML an event
- uses Origin- and FocalMechanismReferences to associate Origins and
  FocalMechanisms,
- Picks are associated via the Arrivals of an Origin and
- Amplitudes are connected via StationMagnitudes of an Origin

Some elements are mandatory in SCML but aren't in QuakeML:
- event/description/type
- dataUsed/stationCount
- dataUsed/componentCount
- amplitude/type


Renaming of nodes
`````````````````

The following table lists the mapping of names between both schema:

Parent              QuakeML name                SCML name
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
seiscomp            eventParameters             EventParameters
arrival             [value copied from fields   weight
                    below in order of listing]
                    timeWeight                  timeUsed (bool)
                    horizontalSlownessWeight    horizontalSlownessUsed (bool)
                    backazimuthWeight           backazimuthUsed (bool)
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
ellipsoid, SCML uses kilometer.


Unmapped node
`````````````

The following nodes can not be mapped to the SCML schema, thus their data is
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

Unlike SCML, QuakeML nodes can appear in any order. They must be reordered for
SCML. Unnecessary attributes must also be removed.

Missing of mandatory elements - Shortcoming of QuakeML validation
`````````````````````````````````````````````````````````````````

Some elements are marked as mandatory in QuakeML (minOccurs=1) but since they
are defined in a xs:choice collection schema validators fail to detect the
absence of such mandatory elements. E.g., it is possible to produce a valid
QuakeML document containing an arrival without a phase definition.


Change log
==========

* 16.06.2021: Add ID_PREFIX parameter allowing to strip QuakeML ID prefix from
  publicIDs and references thereof.

* 22.06.2021: Add Z suffix to xs:dateTime values.

* 18.01.2023:
  - Add GUESS_MAG_ORIGIN switch allowing to map magnitudes to origins
    without an originID reference.
  - Add BUILD_EVENT_ID switch to construct event publicID from
    - last path component of the URI only or
    - by concatenating catalog:eventsource and catalog:eventid attribute as
      used, e.g., by USGS
  - In the absence of a stationSagnitude/waveformID the waveformID will be copied
    from the referenced amplitude if any.
  - Add EVENT_INFO_ID switch which creates unique publicIDs by appending the
    event publicID and event creation time.
  - Add value of '0' for unset dataUsed/stationCount|componentCount.
  - In the absence of the mandatory arrival/phase element phaseHint of the referenced
    pick is used. If neither a phaseHint is available the phase is set to an empty
    value.
  - Remove ID_PREFIX from comment/id and waveformID/resourceURI

-->
<xsl:stylesheet version="1.0"
        xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
        xmlns:xs="http://www.w3.org/2001/XMLSchema"
        xmlns:ext="http://exslt.org/common"
        xmlns="http://geofon.gfz-potsdam.de/ns/seiscomp3-schema/0.8"
        xmlns:qml="http://quakeml.org/xmlns/bed/1.2"
        xmlns:q="http://quakeml.org/xmlns/quakeml/1.2"
        xmlns:catalog="http://anss.org/xmlns/catalog/0.1"
        exclude-result-prefixes="xsl xs ext q qml catalog">
    <xsl:output method="xml" encoding="UTF-8" indent="yes"/>
    <xsl:strip-space elements="*"/>

    <!--
         Define parameters which may be passed to this script.
    -->

    <!-- Prefix to be removed from any publicID -->
    <xsl:param name="ID_PREFIX" select="'smi:org.gfz-potsdam.de/geofon/'"/>

    <!-- PublicID representing an empty value -->
    <xsl:param name="ID_PREFIX_NA" select="concat($ID_PREFIX, 'NA')"/>

    <!-- In QuakeML a magnitude references to its origin via the originID
         element. Some catalogs don't set this element. The GUESS_MAG_ORIGIN
         variable represents a bit mask with certain guessing strategies:
         0: Don't guess the origin.
         1: Single origin. If the document contains only one origin all
            magnitudes and stationMagnitudes without an originID reference will
            be mapped to the single origin.
         2: Preferred origin. Map magnitutes and stationMagnitudes without an
            origin reference to the preferredOrigin. -->
    <xsl:param name="GUESS_MAG_ORIGIN" select="0"/>

    <!-- Special rules for the contruction of the event publicID:
         0: Use ID as is but remove ID_PREFIX as done with any other publicIDs
         1: Extract only the last path component of the URI
         2: Combine event attributes catalog:eventsource and catalog:eventid.
            E.g., this option is suitable for the USGS QuakeML flavor.
    -->
    <xsl:param name="BUILD_EVENT_ID" select="0"/>

    <!-- Make all public IDs unique by appending event publicID and the event
         creation time -->
    <xsl:param name="EVENT_INFO_ID" select="0"/>

    <!-- Define some global variables -->
    <xsl:variable name="version" select="'0.8'"/>
    <xsl:variable name="schema" select="document('sc3ml_0.8.xsd')"/>
    <xsl:variable name="PID" select="'publicID'"/>

    <!-- Define key to remove duplicates-->
    <xsl:key name="pick_key" match="qml:pick" use="@publicID"/>
    <xsl:key name="amplitude_key" match="qml:amplitude" use="@publicID"/>
    <xsl:key name="origin_key" match="qml:origin" use="@publicID"/>
    <xsl:key name="focalMechanism_key" match="qml:focalMechanism" use="@publicID"/>


<!--
    ***************************************************************************
    Utility functions
    ***************************************************************************
-->

    <!-- Reverse seach $delimiter in $string. 'tokenize' only available with
         XSLT 2.0 -->
    <xsl:template name="substring-after-last">
        <xsl:param name="string"/>
        <xsl:param name="delimiter"/>

        <!-- Extract the string which comes after the first occurence -->
        <xsl:variable name="remainder" select="substring-after($string, $delimiter)"/>

        <xsl:choose>
             <!-- If it still contains the search string the recursively process -->
             <xsl:when test="$delimiter and contains($remainder, $delimiter)">
                  <xsl:call-template name="substring-after-last">
                       <xsl:with-param name="string" select="$remainder"/>
                       <xsl:with-param name="delimiter" select="$delimiter"/>
                  </xsl:call-template>
             </xsl:when>
             <xsl:otherwise>
                  <xsl:value-of select="$remainder"/>
             </xsl:otherwise>
        </xsl:choose>
    </xsl:template>

<!--
    ***************************************************************************
    Move/create nodes
    ***************************************************************************
-->

    <!-- Default match: Map node 1:1 -->
    <xsl:template match="*">
        <xsl:element name="{local-name()}">
            <xsl:apply-templates select="@*"/>
            <xsl:apply-templates select="node()"/>
        </xsl:element>
    </xsl:template>

    <!-- Starting point: Match the root node and select the EventParameters
         node -->
    <xsl:template match="/">
        <!-- Write a disordered SCML in this variable. It will be ordered in
             a second run. -->
        <xsl:variable name="disordered">
            <xsl:for-each select="./q:quakeml/qml:eventParameters">
                <EventParameters>
                    <xsl:attribute name="{$PID}">
                        <xsl:call-template name="removeIDPrefix">
                            <xsl:with-param name="id" select="@publicID"/>
                        </xsl:call-template>
                    </xsl:attribute>
                    <xsl:apply-templates select="qml:event"/>
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
            <!-- Special rules for the construction of the event publicID -->
            <xsl:attribute name="{$PID}">
                <xsl:call-template name="eventID">
                    <xsl:with-param name="event" select="current()"/>
                </xsl:call-template>
            </xsl:attribute>
            <xsl:apply-templates/>

            <!-- Create origin references -->
            <xsl:for-each select="qml:origin[count(. | key('origin_key', @publicID)[1]) = 1]">
                <xsl:element name="originReference">
                    <xsl:call-template name="convertID">
                        <xsl:with-param name="id" select="@publicID"/>
                    </xsl:call-template>
                </xsl:element>
            </xsl:for-each>

            <!-- Create focal mechanism references -->
            <xsl:for-each select="qml:focalMechanism[count(. | key('focalMechanism_key', @publicID)[1]) = 1]">
                <xsl:element name="focalMechanismReference">
                    <xsl:call-template name="convertID">
                        <xsl:with-param name="id" select="@publicID"/>
                    </xsl:call-template>
                </xsl:element>
            </xsl:for-each>
        </xsl:element>

        <!-- Copy picks and remove duplicates -->
        <xsl:for-each select="qml:pick[count(. | key('pick_key', @publicID)[1]) = 1]">
            <xsl:element name="{local-name()}">
                <xsl:apply-templates select="@*"/>
                <xsl:apply-templates/>
            </xsl:element>
        </xsl:for-each>

        <!-- Copy amplitudes and remove duplicates -->
        <xsl:for-each select="qml:amplitude[count(. | key('amplitude_key', @publicID)[1]) = 1]">
            <xsl:element name="{local-name()}">
                <xsl:apply-templates select="@*"/>
                <xsl:if test="not(qml:type)">
                    <xsl:element name="type"/>
                </xsl:if>
                <xsl:apply-templates/>
            </xsl:element>
        </xsl:for-each>

        <!-- Definition of fallback origin if GUESS_MAG_ORIGIN is set to a
             value greater than 0. -->
        <xsl:variable name="fallback_origin_id">
            <xsl:choose>
                <!-- Single origin -->
                <xsl:when test="floor($GUESS_MAG_ORIGIN div 1) mod 2 = 1 and count(qml:origin) = 1">
                    <xsl:value-of select="qml:origin[1]/@publicID"/>
                </xsl:when>
                <!-- Preferred origin -->
                <xsl:when test="floor($GUESS_MAG_ORIGIN div 2) mod 2 = 1">
                    <xsl:value-of select="qml:preferredOriginID/text()"/>
                </xsl:when>
            </xsl:choose>
        </xsl:variable>

        <!-- Copy origins and remove duplicates -->
        <xsl:for-each select="qml:origin[count(. | key('origin_key', @publicID)[1]) = 1]">
            <xsl:element name="{local-name()}">
                <xsl:apply-templates select="@*"/>

                <!-- Copy magnitudes -->
                <xsl:for-each select="../qml:magnitude[qml:originID/text()=current()/@publicID or (
                                      not(qml:originID) and current()/@publicID = $fallback_origin_id)]">
                    <xsl:element name="{local-name()}">
                        <xsl:apply-templates select="@*"/>
                        <xsl:apply-templates/>
                    </xsl:element>
                </xsl:for-each>

                <!-- Copy stations magnitudes -->
                <xsl:for-each select="../qml:stationMagnitude[qml:originID/text()=current()/@publicID or (
                                      not(qml:originID) and current()/@publicID = $fallback_origin_id)]">
                    <xsl:element name="{local-name()}">
                        <xsl:apply-templates select="@*"/>
                        <xsl:apply-templates/>

                        <!-- use waveformID of referenced amplitude as fallback -->
                        <xsl:if test="not(qml:waveformID)">
                            <xsl:apply-templates select="../qml:amplitude[current()/qml:amplitudeID/text()=@publicID][1]/qml:waveformID"/>
                        </xsl:if>
                    </xsl:element>
                </xsl:for-each>

                <xsl:apply-templates/>
            </xsl:element>
        </xsl:for-each>

        <!-- Copy focal mechanisms and remove duplicates -->
        <xsl:for-each select="qml:focalMechanism[count(. | key('focalMechanism_key', @publicID)[1]) = 1]">
            <xsl:element name="{local-name()}">
                <xsl:apply-templates select="@*"/>
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
                <xsl:element name="stationCount">
                    <xsl:value-of select="0"/>
                </xsl:element>
            </xsl:if>
            <xsl:if test="not(qml:componentCount)">
                <xsl:element name="componentCount">
                    <xsl:value-of select="0"/>
                </xsl:element>
            </xsl:if>
            <xsl:apply-templates/>
        </xsl:element>
    </xsl:template>

    <!-- Create mandatory element arrival/phase -->
    <xsl:template match="qml:arrival">
        <xsl:element name="{local-name()}">
            <xsl:if test="not(qml:phase)">
                <xsl:element name="phase">
                    <xsl:value-of select="../../qml:pick[current()/qml:pickID/text()=@publicID]/qml:phaseHint/text()"/>
                </xsl:element>
            </xsl:if>
            <xsl:apply-templates/>
        </xsl:element>
    </xsl:template>


<!--
    ***************************************************************************
    Rename nodes/attributes
    ***************************************************************************
-->

    <!-- Manage arrival/weight -->
    <xsl:template match="qml:arrival/qml:timeWeight">
        <xsl:element name="weight">
            <xsl:apply-templates/>
        </xsl:element>
        <xsl:element name="timeUsed">true</xsl:element>
    </xsl:template>
    <xsl:template match="qml:arrival/qml:horizontalSlownessWeight">
        <xsl:if test="not(../qml:timeWeight)">
            <xsl:element name="weight">
                <xsl:apply-templates/>
            </xsl:element>
            <xsl:element name="horizontalSlownessUsed">true</xsl:element>
        </xsl:if>
    </xsl:template>
    <xsl:template match="qml:arrival/qml:backazimuthWeight">
        <xsl:if test="not(../qml:timeWeight) and not(../qml:horizontalSlownessWeight)">
            <xsl:element name="weight">
                <xsl:apply-templates/>
            </xsl:element>
            <xsl:element name="backazimuthUsed">true</xsl:element>
        </xsl:if>
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
            <xsl:apply-templates select="@*"/>
            <xsl:apply-templates/>
        </xsl:element>
    </xsl:template>

    <!-- origin/originUncertainty -> origin/uncertainty -->
    <xsl:template match="qml:origin/qml:originUncertainty">
        <xsl:element name="uncertainty">
            <xsl:apply-templates select="@*"/>
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
                    <xsl:call-template name="removeIDPrefix">
                        <xsl:with-param name="id" select="@id"/>
                    </xsl:call-template>
                </xsl:element>
            </xsl:if>
            <xsl:apply-templates/>
        </xsl:element>
    </xsl:template>

    <!-- waveformID/text() -> waveformID/resourceURI -->
    <xsl:template match="qml:waveformID">
        <xsl:element name="{local-name()}">
            <xsl:apply-templates select="@*"/>
            <xsl:if test="current() != ''">
                <xsl:element name="resourceURI">
                    <xsl:call-template name="removeIDPrefix">
                        <xsl:with-param name="id" select="."/>
                    </xsl:call-template>
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
                <xsl:when test="$v='other event'">other</xsl:when>
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

    <!-- Origin depth, SCML uses kilometer, QuakeML meter -->
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
    Time conversion
    ***************************************************************************
-->

    <!-- SeisComP < 5 requires date time values to end on Z -->
    <xsl:template match="qml:time/qml:value
                        | qml:scalingTime/qml:value
                        | qml:timeWindow/qml:reference
                        | qml:creationTime">
        <xsl:element name="{local-name()}">
            <xsl:variable name="v" select="current()"/>
            <xsl:choose>
                <xsl:when test="substring($v, string-length($v))='Z'">
                    <xsl:value-of select="$v"/>
                </xsl:when>
                <xsl:otherwise>
                    <xsl:value-of select="concat($v, 'Z')"/>
                </xsl:otherwise>
            </xsl:choose>
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

    <!-- Converts a publicID -->
    <xsl:template name="convertID">
        <xsl:param name="id"/>
        <xsl:variable name="res">
            <xsl:call-template name="removeIDPrefix">
                <xsl:with-param name="id" select="$id"/>
            </xsl:call-template>
        </xsl:variable>
        <xsl:choose>
            <xsl:when test="$EVENT_INFO_ID != 0">
                <xsl:call-template name="eventInfoID">
                    <xsl:with-param name="id" select="$res"/>
                </xsl:call-template>
            </xsl:when>
            <xsl:otherwise>
                <xsl:value-of select="$res"/>
            </xsl:otherwise>
        </xsl:choose>
    </xsl:template>

    <!-- Removes ID_PREFIX, if the remainder is 'NA' an empty string is returned -->
    <xsl:template name="removeIDPrefix">
        <xsl:param name="id"/>
        <xsl:choose>
            <xsl:when test="$id=$ID_PREFIX_NA">
                <xsl:value-of select="''"/>
            </xsl:when>
            <xsl:when test="starts-with($id, $ID_PREFIX)">
                <xsl:value-of select="substring-after($id, $ID_PREFIX)"/>
            </xsl:when>
            <xsl:otherwise>
                <xsl:value-of select="$id"/>
            </xsl:otherwise>
        </xsl:choose>
    </xsl:template>

    <!-- Special rules for the construction of the event publicID -->
    <xsl:template name="eventID">
        <xsl:param name="event"/>
        <xsl:choose>
            <!-- Use last id path component -->
            <xsl:when test="$BUILD_EVENT_ID = 1">
                <xsl:call-template name="substring-after-last">
                    <xsl:with-param name="string" select="$event/@publicID"/>
                    <xsl:with-param name="delimiter" select="'/'"/>
                </xsl:call-template>
            </xsl:when>
            <!-- Use catalog:eventsource and catalog:eventid -->
            <xsl:when test="$BUILD_EVENT_ID = 2">
                <xsl:value-of select="concat($event/@catalog:eventsource, $event/@catalog:eventid)"/>
            </xsl:when>
            <xsl:otherwise>
                <xsl:call-template name="removeIDPrefix">
                    <xsl:with-param name="id" select="$event/@publicID"/>
                </xsl:call-template>
            </xsl:otherwise>
        </xsl:choose>
    </xsl:template>

    <!-- Adds eventID and event creation time to id parameter -->
    <xsl:template name="eventInfoID">
        <xsl:param name="id"/>
        <xsl:variable name="event" select="ancestor::*[last()-2]"/>
        <xsl:variable name="eventID">
            <xsl:call-template name="eventID">
                <xsl:with-param name="event" select="$event"/>
            </xsl:call-template>
        </xsl:variable>
        <xsl:value-of select="concat($id, '/', $eventID, '/', $event/qml:creationInfo/qml:creationTime)"/>
    </xsl:template>

    <!-- Remove ID_PREFIX from publicID attributes -->
    <xsl:template match="@publicID">
        <xsl:variable name="id">
            <xsl:call-template name="convertID">
                <xsl:with-param name="id" select="current()"/>
            </xsl:call-template>
        </xsl:variable>
        <xsl:if test="$id != ''">
            <xsl:attribute name="{$PID}">
                <xsl:value-of select="$id"/>
            </xsl:attribute>
        </xsl:if>
    </xsl:template>

    <!-- Generic template for all remaining attributes -->
    <xsl:template match="@*">
        <xsl:copy-of select="."/>
    </xsl:template>

    <!-- ID nodes which must be stripped from ID_PREFIX -->
    <xsl:template match="qml:agencyURI
                       | qml:authorURI
                       | qml:greensFunctionID
                       | qml:filterID
                       | qml:methodID
                       | qml:earthModelID
                       | qml:referenceSystemID
                       | qml:slownessMethodID">
        <xsl:variable name="id">
            <xsl:call-template name="removeIDPrefix">
                <xsl:with-param name="id" select="string(.)"/>
            </xsl:call-template>
        </xsl:variable>
        <xsl:if test="$id != ''">
            <xsl:element name="{local-name()}">
                <xsl:value-of select="$id"/>
            </xsl:element>
        </xsl:if>
    </xsl:template>

    <!-- ID nodes referencing public objects: Remove ID_PREFIX and optionally ensure ID uniqueness. -->
    <xsl:template match="qml:derivedOriginID
                       | qml:momentMagnitudeID
                       | qml:triggeringOriginID
                       | qml:pickID
                       | qml:stationMagnitudeID
                       | qml:originID
                       | qml:amplitudeID
                       | qml:preferredOriginID
                       | qml:preferredMagnitudeID
                       | qml:preferredFocalMechanismID">
        <xsl:variable name="id">
            <xsl:call-template name="convertID">
                <xsl:with-param name="id" select="string(.)"/>
            </xsl:call-template>
        </xsl:variable>
        <xsl:if test="$id != ''">
            <xsl:element name="{local-name()}">
                <xsl:value-of select="$id"/>
            </xsl:element>
        </xsl:if>
    </xsl:template>

</xsl:stylesheet>
