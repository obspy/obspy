<?xml version="1.0" encoding="UTF-8"?>
<!-- **********************************************************************
 * Copyright (C) 2017 by gempa GmbH
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Lesser Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 *
 * SC3ML 0.11 to QuakeML 1.2 stylesheet converter
 * Author  : Stephan Herrnkind
 * Email   : stephan.herrnkind@gempa.de
 * Version : 2017.342.01
 *
 * ================
 * Usage
 * ================
 *
 * This stylesheet converts a SC3ML to a QuakeML document. It may be invoked
 * e.g. using xalan or xsltproc:
 *
 *   xalan -in sc3ml.xml -xsl sc3ml_0.11__quakeml_1.2.xsl -out quakeml.xml
 *   xsltproc -o quakeml.xml sc3ml_0.11__quakeml_1.2.xsl sc3ml.xml
 *
 * You can also modify the default ID prefix with the reverse DNS name of your
 * institute by setting the ID_PREFIX param:
 *
 *   xalan -param ID_PREFIX "'smi:org.gfz-potsdam.de/geofon/'" -in sc3ml.xml -xsl sc3ml_0.11__quakeml_1.2.xsl -out quakeml.xml
 *   xsltproc -stringparam ID_PREFIX smi:org.gfz-potsdam.de/geofon/ -o quakeml.xml sc3ml_0.11__quakeml_1.2.xsl sc3ml.xml
 *
 * ================
 * Transformation
 * ================
 *
 * QuakeML and SC3ML are quite similar schemas. Nevertheless some differences
 * exist:
 *
 *  - IDs : SC3ML does not enforce any particular ID restriction. An ID in
 *    SC3ML has no semantic, it simply must be unique. Hence QuakeML uses ID
 *    restrictions, a conversion of a SC3ML to a QuakeML ID must be performed:
 *    'sc3id' -> 'smi:org.gfz-potsdam.de/geofon/'. If no SC3ML ID is available
 *    but QuakeML enforces one, a static ID value of 'NA' is used.
 *    If the ID starts with `smi:` or `quakeml:`, the ID is considered valid
 *    and let untouched. This can lead to an invalid generated file but avoid
 *    to always modify IDs, especially when converting several times.
 *  - Repositioning of nodes: In QuakeML all information is grouped under the
 *    event element. As a consequence every node not referenced by an event
 *    will be lost during the conversion.
 *
 *    <EventParameters>               <eventParameters>
 *                                        <event>
 *        <pick/>                             <pick/>
 *        <amplitude/>                        <amplitude/>
 *        <reading/>
 *        <origin>                            <origin/>
 *            <stationMagnitude/>             <stationMagnitude/>
 *            <magnitude/>                    <magnitude/>
 *        </origin>
 *        <focalMechanism/>                   <focalMechanism/>
 *        <event/>                        </event>
 *    </EventParameters>              </eventParameters>
 *
 *  - Renaming of nodes: The following table lists the mapping of names between
 *    both schema:
 *
 *    Parent (SC3)     SC3 name                 QuakeML name
 *    """""""""""""""""""""""""""""""""""""""""""""""""""""""
 *    seiscomp         EventParameters          eventParameters
 *    arrival          weight [copied to following fields if true]
 *                       timeUsed               timeWeight
 *                       horizontalSlownessUsed horizontalSlownessWeight
 *                       backazimuthUsed        backazimuthWeight
 *                     takeOffAngle             takeoffAngle
 *    magnitude        magnitude                mag
 *    stationMagnitude magnitude                mag
 *    amplitude        amplitude                genericAmplitude
 *    origin           uncertainty              originUncertainty
 *    momentTensor     method                   category
 *    waveformID       resourceURI              CDATA
 *    comment          id                       id (attribute)
 *
 *  - Enumerations: Both schema use enumerations. Numerous mappings are applied.
 *
 *  - Unit conversion: SC3ML uses kilometer for origin depth, origin
 *    uncertainty and confidence ellipsoid, QuakeML uses meter
 *
 *  - Unmapped nodes: The following nodes can not be mapped to the QuakeML
 *    schema, thus their data is lost:
 *
 *    Parent           Element lost
 *    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
 *    creationInfo     modificationTime
 *    momentTensor     method
 *                     stationMomentTensorContribution
 *                     status
 *                     cmtName
 *                     cmtVersion
 *                     phaseSetting
 *    stationMagnitude passedQC
 *    eventParameters  reading
 *    comment          start
 *    comment          end
 *    RealQuantity     pdf
 *    TimeQuality      pdf
 *
 *  - Mandatory nodes: The following nodes is mandatory in QuakeML but not in
 *    SC3ML:
 *
 *    Parent           Mandatory element
 *    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""
 *    Amplitude        genericAmplitude
 *    StationMagnitude originID
 *
 *  - Restriction of data size: QuakeML restricts string length of some
 *    elements. This length restriction is _NOT_ enforced by this
 *    stylesheet to prevent data loss. As a consequence QuakeML files
 *    generated by this XSLT may not validate because of these
 *    restrictions.
 *
 * ================
 * Change log
 * ===============
 *
 *  * 08.09.2014: Fixed typo in event type conversion (meteo[r] impact)
 *
 *  * 25.08.2014: Applied part of the patch proposed by Philipp Kaestli on
 *                seiscomp-l@gfz-potsdam.de
 *    - use public id of parent origin if origin id propertery of magnitude
 *      and station magnitude elements is unset
 *    - fixed takeOffAngle conversion vom real (SC3ML) to RealQuantity
 *      (QuakeML)
 *
 *  * 04.07.2016: Version bump. No modification here, SC3 datamodel was updated
 *                on the inventory side.
 *
 *  * 28.11.2016: Version bump. No modification here, SC3 datamodel was updated
 *                on the inventory side.
 *
 *  * 28.06.2017: Changed license from GPL to LGPL
 *
 *  * 08.08.2017: Added some fixes to use this XSLT in ObsPy
 *    - Change ID_PREFIX variable to a param
 *    - Do not copy Amplitude if amplitude/amplitude doesn't existing
 *    - focalMechanism/evaluationMode and focalMechanism/evaluationStatus were
 *      not copied but can actually be mapped to the QuakeML schema
 *    - Some event/type enumeration was mapped to `other` instead of
 *      `other event`
 *    - Fix origin uncertainty and confidence ellispoid units
 *    - Rename momentTensor/method to momentTensor/category
 *    - Fix amplitude/unit (enumeration in QuakeML, not in SC3ML)
 *    - Don't modify id if it starts with 'smi:' or 'quakeml:'
 *    - Fix Arrival publicID generation
 *
 *  * 27.09.2017:
 *    - Use '_' instead of '#' in arrival publicID generation
 *    - Map SC3 arrival weight to timeWeight, horizontalSlownessWeight and
 *      backazimuthWeight depending on timeUsed, horizontalUsed and
 *      backzimuthUsed values
 *
 *  * 08.12.2017:
 *    - Remove unmapped nodes
 *    - Fix arrival weight mapping
 *
 *  * 27.07.2018: Version bump. No modification here, SC3 datamodel was
 *                extented by data availability top level element
 *
 *  * 02.11.2018: Don't export stationMagnitude passedQC attribute
 *
 *  * 07.12.2018: Copy picks referenced by amplitudes
 *
 *  * 10.12.2018: Put the non-QuakeML nodes in a custom namespace
 *
 ********************************************************************** -->
<xsl:stylesheet version="1.0"
        xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
        xmlns:scs="http://geofon.gfz-potsdam.de/ns/seiscomp3-schema/0.11"
        xmlns:qml="http://quakeml.org/xmlns/quakeml/1.0"
        xmlns="http://quakeml.org/xmlns/bed/1.2"
        xmlns:q="http://quakeml.org/xmlns/quakeml/1.2"
        exclude-result-prefixes="scs qml xsl">
    <xsl:output method="xml" encoding="UTF-8" indent="yes"/>
    <xsl:strip-space elements="*"/>

    <!-- Define parameters-->
    <xsl:param name="ID_PREFIX" select="'smi:org.gfz-potsdam.de/geofon/'"/>

    <!-- Define global variables -->
    <xsl:variable name="PID" select="'publicID'"/>

    <!-- Starting point: Match the root node and select the one and only
         EventParameters node -->
    <xsl:template match="/">
        <xsl:variable name="scsRoot" select="."/>
        <q:quakeml>
            <xsl:for-each select="$scsRoot/scs:seiscomp/scs:EventParameters">
                <eventParameters>
                    <!-- Mandatory publicID attribute -->
                    <xsl:attribute name="{$PID}">
                        <xsl:call-template name="convertOptionalID">
                            <xsl:with-param name="id" select="@publicID"/>
                        </xsl:call-template>
                    </xsl:attribute>

                    <!-- Put the QuakeML nodes at the beginning -->
                    <xsl:apply-templates select="*[not(self::scs:reading)]" />
                    <!-- Put the non-QuakeML nodes at the end -->
                    <xsl:apply-templates select="scs:reading" mode="scs-only" />
                </eventParameters>
            </xsl:for-each>
        </q:quakeml>
    </xsl:template>

    <!-- event -->
    <xsl:template match="scs:event">
        <xsl:element name="{local-name()}">
            <xsl:apply-templates select="@*"/>

            <!-- search origins referenced by this event -->
            <xsl:for-each select="scs:originReference">
                <xsl:for-each select="../../scs:origin[@publicID=current()]">
                    <xsl:variable name="origin" select="current()" />

                    <!-- stationMagnitudes and referenced amplitudes -->
                    <xsl:for-each select="scs:stationMagnitude">
                        <xsl:for-each select="../../scs:amplitude[@publicID=current()/scs:amplitudeID]">
                            <!-- amplitude/genericAmplitude is mandatory in QuakeML -->
                            <xsl:if test="scs:amplitude">
                                <!-- copy picks referenced in amplitudes -->
                                <xsl:for-each select="../scs:pick[@publicID=current()/scs:pickID]">
                                    <xsl:call-template name="genericNode" />
                                </xsl:for-each>
                                <xsl:call-template name="genericNode"/>
                            </xsl:if>
                        </xsl:for-each>
                        <xsl:apply-templates select="." mode="originMagnitude">
                            <xsl:with-param name="oID" select="../@publicID"/>
                        </xsl:apply-templates>
                    </xsl:for-each>

                    <!-- magnitudes -->
                    <xsl:for-each select="scs:magnitude">
                        <xsl:apply-templates select="." mode="originMagnitude">
                            <xsl:with-param name="oID" select="../@publicID"/>
                        </xsl:apply-templates>
                    </xsl:for-each>

                    <!-- picks, referenced by arrivals -->
                    <xsl:for-each select="scs:arrival">
                        <!--xsl:value-of select="scs:pickID"/-->
                        <!-- Don't copy picks already referenced in amplitudes -->
                        <xsl:for-each select="
                                ../../scs:pick[
                                    @publicID=current()/scs:pickID
                                    and not(@publicID=../scs:amplitude[
                                        @publicID=$origin/scs:stationMagnitude/scs:amplitudeID]/scs:pickID)]">
                            <xsl:call-template name="genericNode"/>
                        </xsl:for-each>
                    </xsl:for-each>

                    <!-- origin -->
                    <xsl:call-template name="genericNode"/>
                </xsl:for-each>
            </xsl:for-each>

            <!-- search focalMechanisms referenced by this event -->
            <xsl:for-each select="scs:focalMechanismReference">
                <xsl:for-each select="../../scs:focalMechanism[@publicID=current()]">
                    <xsl:call-template name="genericNode"/>
                </xsl:for-each>
            </xsl:for-each>

            <xsl:apply-templates/>
        </xsl:element>
    </xsl:template>

    <!-- Default match: Map node 1:1 -->
    <xsl:template match="*">
        <xsl:call-template name="genericNode"/>
    </xsl:template>

    <!-- Delete elements -->
    <xsl:template match="scs:EventParameters/scs:pick"/>
    <xsl:template match="scs:EventParameters/scs:amplitude"/>
    <xsl:template match="scs:EventParameters/scs:origin"/>
    <xsl:template match="scs:EventParameters/scs:focalMechanism"/>
    <xsl:template match="scs:event/scs:originReference"/>
    <xsl:template match="scs:event/scs:focalMechanismReference"/>
    <xsl:template match="scs:comment/scs:id"/>
    <xsl:template match="scs:arrival/scs:weight"/>
    <xsl:template match="scs:arrival/scs:timeUsed"/>
    <xsl:template match="scs:arrival/scs:horizontalSlownessUsed"/>
    <xsl:template match="scs:arrival/scs:backazimuthUsed"/>
    <xsl:template match="scs:origin/scs:stationMagnitude"/>
    <xsl:template match="scs:origin/scs:magnitude"/>
    <xsl:template match="scs:momentTensor/scs:method"/>

    <!-- Converts a scs magnitude/stationMagnitude to a qml
         magnitude/stationMagnitude -->
    <xsl:template match="*" mode="originMagnitude">
        <xsl:param name="oID"/>
        <xsl:element name="{local-name()}">
            <xsl:apply-templates select="@*"/>
            <!-- if no originID element is available, create one with
                 the value of the publicID attribute of parent origin -->
            <xsl:if test="not(scs:originID)">
                <originID>
                    <xsl:call-template name="convertID">
                        <xsl:with-param name="id" select="$oID"/>
                    </xsl:call-template>
                </originID>
            </xsl:if>

            <!-- Put the QuakeML nodes at the beginning -->
            <xsl:apply-templates select="*[not(self::scs:passedQC)]" />
            <!-- Put the non-QuakeML nodes at the end -->
            <xsl:apply-templates select="scs:passedQC" mode="scs-only" />
        </xsl:element>
    </xsl:template>

    <!-- event type, enumeration differs slightly -->
    <xsl:template match="scs:event/scs:type">
        <xsl:element name="{local-name()}">
            <xsl:variable name="v" select="current()"/>
            <xsl:choose>
                <xsl:when test="$v='induced earthquake'">induced or triggered event</xsl:when>
                <xsl:when test="$v='meteor impact'">meteorite</xsl:when>
                <xsl:when test="$v='not locatable'">other event</xsl:when>
                <xsl:when test="$v='outside of network interest'">other event</xsl:when>
                <xsl:when test="$v='duplicate'">other event</xsl:when>
                <xsl:when test="$v='other'">other event</xsl:when>
                <xsl:otherwise><xsl:value-of select="$v"/></xsl:otherwise>
            </xsl:choose>
        </xsl:element>
    </xsl:template>

    <!-- origin depth, SC3ML uses kilometer, QML meter -->
    <xsl:template match="scs:origin/scs:depth/scs:value
                         | scs:origin/scs:depth/scs:uncertainty
                         | scs:origin/scs:depth/scs:lowerUncertainty
                         | scs:origin/scs:depth/scs:upperUncertainty
                         | scs:origin/scs:uncertainty/scs:horizontalUncertainty
                         | scs:origin/scs:uncertainty/scs:minHorizontalUncertainty
                         | scs:origin/scs:uncertainty/scs:maxHorizontalUncertainty
                         | scs:confidenceEllipsoid/scs:semiMajorAxisLength
                         | scs:confidenceEllipsoid/scs:semiMinorAxisLength
                         | scs:confidenceEllipsoid/scs:semiIntermediateAxisLength">
        <xsl:element name="{local-name()}">
            <xsl:value-of select="current() * 1000"/>
        </xsl:element>
    </xsl:template>

    <!-- evaluation status, enumeration of QML does not include 'reported' -->
    <xsl:template match="scs:evaluationStatus">
        <xsl:variable name="v" select="current()"/>
        <xsl:if test="$v!='reported'">
            <xsl:element name="{local-name()}">
                <xsl:value-of select="$v"/>
            </xsl:element>
        </xsl:if>
    </xsl:template>

    <!-- data used wave type, enumeration differs slightly -->
    <xsl:template match="scs:dataUsed/scs:waveType">
        <xsl:element name="{local-name()}">
            <xsl:variable name="v" select="current()"/>
            <xsl:choose>
                <xsl:when test="$v='P body waves'">P waves</xsl:when>
                <xsl:when test="$v='long-period body waves'">body waves</xsl:when>
                <xsl:when test="$v='intermediate-period surface waves'">surface waves</xsl:when>
                <xsl:when test="$v='long-period mantle waves'">mantle waves</xsl:when>
                <xsl:otherwise><xsl:value-of select="$v"/></xsl:otherwise>
            </xsl:choose>
        </xsl:element>
    </xsl:template>

    <!-- origin uncertainty description, enumeration of QML does not include 'probability density function' -->
    <xsl:template match="scs:origin/scs:uncertainty/scs:preferredDescription">
        <xsl:variable name="v" select="current()"/>
        <xsl:if test="$v!='probability density function'">
            <xsl:element name="{local-name()}">
                <xsl:value-of select="$v"/>
            </xsl:element>
        </xsl:if>
    </xsl:template>

    <!-- momentTensor/method -> momentTensor/category -->
    <xsl:template match="scs:momentTensor/scs:method">
        <xsl:variable name="v" select="current()"/>
        <xsl:if test="$v='teleseismic' or $v='regional'">
            <xsl:element name="category">
                <xsl:value-of select="$v"/>
            </xsl:element>
        </xsl:if>
    </xsl:template>

    <!-- amplitude/unit is an enumeration in QuakeML, not in SC3ML -->
    <xsl:template match="scs:amplitude/scs:unit">
        <xsl:variable name="v" select="current()"/>
        <xsl:element name="{local-name()}">
            <xsl:choose>
                <xsl:when test="$v='m'
                                or $v='s'
                                or $v='m/s'
                                or $v='m/(s*s)'
                                or $v='m*s'
                                or $v='dimensionless'">
                    <xsl:value-of select="$v"/>
                </xsl:when>
                <xsl:otherwise>other</xsl:otherwise>
            </xsl:choose>
        </xsl:element>
    </xsl:template>

    <!-- origin arrival -->
    <xsl:template match="scs:arrival">
        <xsl:element name="{local-name()}">
            <!-- since SC3ML does not include a publicID it is generated from pick and origin id -->
            <xsl:attribute name="{$PID}">
                <xsl:call-template name="convertID">
                    <xsl:with-param name="id" select="concat(scs:pickID, '_', translate(../@publicID, ' :', '__'))"/>
                </xsl:call-template>
            </xsl:attribute>
            <!-- mapping of weight to timeWeight, horizontalSlownessWeight and backazimuthWeight
                 depending on timeUsed, horizontalSlownessUsed and backazimuthUsed values -->
            <xsl:choose>
                <xsl:when test="scs:weight">
                    <xsl:if test="((scs:timeUsed='true') or (scs:timeUsed='1'))
                                  or (not(scs:timeUsed|scs:horizontalSlownessUsed|scs:backazimuthUsed))">
                        <xsl:element name="timeWeight">
                            <xsl:value-of select="scs:weight"/>
                        </xsl:element>
                    </xsl:if>
                    <xsl:if test="((scs:horizontalSlownessUsed='true') or (scs:horizontalSlownessUsed='1'))">
                        <xsl:element name="horizontalSlownessWeight">
                            <xsl:value-of select="scs:weight"/>
                        </xsl:element>
                    </xsl:if>
                    <xsl:if test="((scs:backazimuthUsed='true') or (scs:backazimuthUsed='1'))">
                        <xsl:element name="backazimuthWeight">
                            <xsl:value-of select="scs:weight"/>
                        </xsl:element>
                    </xsl:if>
                </xsl:when>
                <xsl:otherwise>
                    <xsl:if test="((scs:timeUsed='true') or (scs:timeUsed='1'))">
                        <xsl:element name="timeWeight">
                            <xsl:value-of select="'1'"/>
                        </xsl:element>
                    </xsl:if>
                    <xsl:if test="((scs:horizontalSlownessUsed='true') or (scs:horizontalSlownessUsed='1'))">
                        <xsl:element name="horizontalSlownessWeight">
                            <xsl:value-of select="'1'"/>
                        </xsl:element>
                    </xsl:if>
                    <xsl:if test="((scs:backazimuthUsed='true') or (scs:backazimuthUsed='1'))">
                        <xsl:element name="backazimuthWeight">
                            <xsl:value-of select="'1'"/>
                        </xsl:element>
                    </xsl:if>
                </xsl:otherwise>
            </xsl:choose>
            <xsl:apply-templates/>
        </xsl:element>
    </xsl:template>

    <!-- Value of ID nodes must be converted to a qml identifier -->
    <xsl:template match="scs:agencyURI|scs:authorURI|scs:pickID|scs:methodID|scs:earthModelID|scs:amplitudeID|scs:originID|scs:stationMagnitudeID|scs:preferredOriginID|scs:preferredMagnitudeID|scs:originReference|scs:filterID|scs:slownessMethodID|scs:pickReference|scs:amplitudeReference|scs:referenceSystemID|scs:triggeringOriginID|scs:derivedOriginID|momentMagnitudeID|scs:preferredFocalMechanismID|scs:focalMechanismReference|scs:momentMagnitudeID|scs:greensFunctionID">
        <xsl:element name="{local-name()}">
            <xsl:apply-templates select="@*"/>
            <xsl:call-template name="valueOfIDNode"/>
        </xsl:element>
    </xsl:template>

    <!-- arrival/takeOffAngle -> arrival/takeoffAngle -->
    <xsl:template match="scs:arrival/scs:takeOffAngle">
        <xsl:element name="takeoffAngle">
            <xsl:element name="value">
                <xsl:value-of select="."/>
            </xsl:element>
        </xsl:element>
    </xsl:template>

    <!-- stationMagnitude/magnitude -> stationMagnitude/mag -->
    <xsl:template match="scs:stationMagnitude/scs:magnitude|scs:magnitude/scs:magnitude">
        <xsl:call-template name="genericNode">
            <xsl:with-param name="name" select="'mag'"/>
        </xsl:call-template>
    </xsl:template>

    <!-- amplitude/amplitude -> amplitude/genericAmplitude -->
    <xsl:template match="scs:amplitude/scs:amplitude">
        <xsl:call-template name="genericNode">
            <xsl:with-param name="name" select="'genericAmplitude'"/>
        </xsl:call-template>
    </xsl:template>

    <!-- origin/uncertainty -> origin/originUncertainty -->
    <xsl:template match="scs:origin/scs:uncertainty">
        <xsl:call-template name="genericNode">
            <xsl:with-param name="name" select="'originUncertainty'"/>
        </xsl:call-template>
    </xsl:template>

    <!-- waveformID: SCS uses a child element 'resourceURI', QML
         inserts the URI directly as value -->
    <xsl:template match="scs:waveformID">
        <xsl:element name="{local-name()}">
            <xsl:apply-templates select="@*"/>
            <xsl:if test="scs:resourceURI">
                <xsl:call-template name="convertID">
                    <xsl:with-param name="id" select="scs:resourceURI"/>
                </xsl:call-template>
            </xsl:if>
        </xsl:element>
    </xsl:template>

    <!-- comment: SCS uses a child element 'id', QML an attribute 'id' -->
    <xsl:template match="scs:comment">
        <xsl:element name="{local-name()}">
            <xsl:if test="scs:id">
                <xsl:attribute name="id">
                    <xsl:call-template name="convertID">
                        <xsl:with-param name="id" select="scs:id"/>
                    </xsl:call-template>
                </xsl:attribute>
            </xsl:if>

            <!-- Put the QuakeML nodes at the beginning -->
            <xsl:apply-templates select="*[not(self::scs:start|self::scs:end)]" />
            <!-- Put the non-QuakeML nodes at the end -->
            <xsl:apply-templates select="scs:start|scs:end" mode="scs-only" />
        </xsl:element>
    </xsl:template>

    <!-- Generic transformation of all attributes of an element. If the
         attribute name is 'eventID' it is transfered to a QML id -->
    <xsl:template match="@*">
        <xsl:variable name="attName" select="local-name()"/>
        <xsl:attribute name="{$attName}">
            <xsl:choose>
                <xsl:when test="$attName=$PID">
                    <xsl:call-template name="convertID">
                        <xsl:with-param name="id" select="string(.)"/>
                    </xsl:call-template>
                </xsl:when>
                <xsl:otherwise>
                    <xsl:value-of select="string(.)"/>
                </xsl:otherwise>
            </xsl:choose>
        </xsl:attribute>
    </xsl:template>

<!--
    ************************************************************************
    Unmapped nodes
    ************************************************************************
-->

    <xsl:template match="scs:creationInfo">
        <xsl:element name="{local-name()}">
            <xsl:apply-templates select="@*"/>

            <!-- Put the QuakeML nodes at the beginning -->
            <xsl:apply-templates select="*[not(self::scs:modificationTime)]" />
            <!-- Put the non-QuakeML nodes at the end -->
            <xsl:apply-templates select="scs:modificationTime" mode="scs-only" />
        </xsl:element>
    </xsl:template>

    <xsl:template match="scs:momentTensor">
        <xsl:element name="{local-name()}">
            <xsl:apply-templates select="@*"/>

            <!-- Put the QuakeML nodes at the beginning -->
            <xsl:apply-templates select="*[not(self::scs:stationMomentTensorContribution
                                               | self::scs:status
                                               | self::scs:cmtName
                                               | self::scs:cmtVersion
                                               | self::scs:phaseSetting)]" />
            <!-- Put the non-QuakeML nodes at the end -->
            <xsl:apply-templates select="scs:stationMomentTensorContribution
                                         | scs:status
                                         | scs:cmtName
                                         | scs:cmtVersion
                                         | scs:phaseSetting" mode="scs-only" />
        </xsl:element>
    </xsl:template>

    <xsl:template match="scs:pdf">
        <xsl:apply-templates select="." mode="scs-only" />
    </xsl:template>

    <xsl:template match="node()|@*" mode="scs-only">
      <xsl:copy>
        <xsl:apply-templates select="node()|@*"/>
      </xsl:copy>
    </xsl:template>

    <!-- Keep seiscomp namespace for unmapped node -->
    <xsl:template match="scs:*" mode="scs-only">
      <xsl:element name="scs:{local-name()}">
        <xsl:apply-templates select="@*|node()" mode="scs-only" />
      </xsl:element>
  </xsl:template>


<!--
    ************************************************************************
    Named Templates
    ************************************************************************
-->

    <!-- Generic and recursively transformation of elements and their
         attributes -->
    <xsl:template name="genericNode">
        <xsl:param name="name"/>
        <xsl:param name="reqPID"/>
        <xsl:variable name="nodeName">
            <xsl:choose>
                <xsl:when test="$name">
                    <xsl:value-of select="$name"/>
                </xsl:when>
                <xsl:otherwise>
                    <xsl:value-of select="local-name()"/>
                </xsl:otherwise>
            </xsl:choose>
        </xsl:variable>
        <xsl:element name="{$nodeName}">
            <xsl:apply-templates select="@*"/>
            <xsl:if test="$reqPID">
                <xsl:attribute name="{$PID}">
                    <xsl:call-template name="convertOptionalID">
                        <xsl:with-param name="id" select="@publicID"/>
                    </xsl:call-template>
                </xsl:attribute>
            </xsl:if>
            <xsl:apply-templates/>
        </xsl:element>
    </xsl:template>

    <!-- Converts and returns value of an id node -->
    <xsl:template name="valueOfIDNode">
        <xsl:call-template name="convertOptionalID">
            <xsl:with-param name="id" select="string(.)"/>
        </xsl:call-template>
    </xsl:template>

    <!-- Converts a scs id to a quakeml id. If the scs id is not set
         the constant 'NA' is used -->
    <xsl:template name="convertOptionalID">
        <xsl:param name="id"/>
        <xsl:choose>
            <xsl:when test="$id">
                <xsl:call-template name="convertID">
                    <xsl:with-param name="id" select="$id"/>
                </xsl:call-template>
            </xsl:when>
            <xsl:otherwise>
                <xsl:call-template name="convertID">
                    <!--xsl:with-param name="id" select="concat('NA-', generate-id())"/-->
                    <xsl:with-param name="id" select="'NA'"/>
                </xsl:call-template>
            </xsl:otherwise>
        </xsl:choose>
    </xsl:template>

    <!-- Converts a scs id to a quakeml id -->
    <xsl:template name="convertID">
        <xsl:param name="id"/>
        <!-- If the id starts with 'smi:' or 'quakeml:', consider that the id
             is already well formated -->
        <xsl:choose>
            <xsl:when test="starts-with($id, 'smi:')
                            or starts-with($id, 'quakeml:')">
                <xsl:value-of select="$id"/>
            </xsl:when>
            <xsl:otherwise>
                <xsl:value-of select="concat($ID_PREFIX, translate($id, ' :', '__'))"/>
            </xsl:otherwise>
        </xsl:choose>
    </xsl:template>

</xsl:stylesheet>
