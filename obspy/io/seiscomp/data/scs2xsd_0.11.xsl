<?xml version="1.0"?>
<!-- **********************************************************************
 * (C) 2006 - GFZ-Potsdam
 *
 * Author: Andres Heinloo
 * Email: geofon_devel@gfz-potsdam.de
 * $Date: 2011-06-09 17:14:47 +0200 (Do, 09. Jun 2011) $
 * $Revision: 6618 $
 * $LastChangedBy: smueller $
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the
 * Free Software Foundation, Inc.,
 * 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 ********************************************************************** -->

<xsl:transform version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
    xmlns:xs="http://www.w3.org/2001/XMLSchema"
    xmlns:scs="http://geofon.gfz-potsdam.de/ns/seiscomp3-schema/0.11">

<xsl:output method="xml" indent="yes"/>
<xsl:variable name="root">seiscomp</xsl:variable>
<xsl:variable name="scs" select="'scs:'"/>

<xsl:key name="elements_parents"
    match="//scs:type"
    use="scs:element/@typeref"/>

<xsl:key name="attributes_parents"
    match="//scs:type"
    use="scs:attribute/@typeref"/>

<xsl:template match="scs:seiscomp-schema">
    <xsl:comment> Generated from Seiscomp Schema, do not edit </xsl:comment>
    <xs:schema targetNamespace="{namespace-uri()}" elementFormDefault="qualified" attributeFormDefault="unqualified">
        <xs:simpleType name="ResourceIdentifier">
            <xs:restriction base="xs:string"/>
        </xs:simpleType>
        <xs:simpleType name="FloatArrayType">
            <xs:list itemType="xs:double"/>
        </xs:simpleType>
        <xs:simpleType name="ComplexArrayType">
            <xs:restriction base="xs:string">
                <xs:pattern value="(\s*\(\s*[+\-]?[0-9]+(\.[0-9]+)?([Ee][+\-][0-9]+)?\s*,\s*[+\-]?[0-9]+(\.[0-9]+)?([Ee][+\-][0-9]+)?\s*\)\s*)*"/>
            </xs:restriction>
        </xs:simpleType>
        <xs:simpleType name="TimeArrayType">
            <xs:list itemType="xs:dateTime"/>
        </xs:simpleType>
        <xsl:apply-templates select="scs:enum"/>
        <xsl:apply-templates select="scs:type"/>
        <xs:element name="{$root}">
            <xs:complexType>
                <xs:all>
                    <xsl:apply-templates select="scs:element" mode="package"/>
                </xs:all>
                <xs:attribute name="version" type="xs:string"/>
            </xs:complexType>
        </xs:element>
    </xs:schema>
</xsl:template>

<xsl:template match="scs:enum">
        <xs:simpleType name="{@name}">
            <xs:restriction base="xs:string">
                <xsl:apply-templates select="scs:option"/>
            </xs:restriction>
        </xs:simpleType>
</xsl:template>
 
<xsl:template match="scs:option">
    <xs:enumeration value="{@value}"/>
</xsl:template>

<xsl:template match="scs:type">
        <xsl:variable name="attribute" select="scs:attribute"/>
        <xsl:variable name="element" select="scs:element"/>
        <xs:complexType name="{@name}">
            <xsl:choose>
                <xsl:when test="$attribute[@xmltype = 'cdata']">
                    <xsl:apply-templates select="$attribute[@xmltype = 'cdata']" mode="cdata"/>
                </xsl:when>
                <xsl:otherwise>
                    <xsl:choose>
                        <xsl:when test="($attribute[@xmltype = 'element']) and ($element)">
                            <xs:sequence>
                                <xsl:apply-templates select="$attribute[@xmltype = 'element']" mode="element"/>
                                <xsl:apply-templates select="$element"/>
                            </xs:sequence>
                        </xsl:when>
                        <xsl:when test="$attribute[@xmltype = 'element']">
                            <xs:sequence>
                                <xsl:apply-templates select="$attribute[@xmltype = 'element']" mode="element"/>
                            </xs:sequence>
                        </xsl:when>
                        <xsl:when test="$element">
                            <xs:sequence>
                                <xsl:apply-templates select="$element"/>
                            </xs:sequence>
                        </xsl:when>
                    </xsl:choose>
                    <xsl:apply-templates select="$attribute[@xmltype = 'attribute']"/>
                </xsl:otherwise>
            </xsl:choose>
        </xs:complexType>
</xsl:template>

<xsl:template match="scs:element" mode="package">
    <!-- Use typeref as name since packages aren't objects -->
    <xs:element name="{@typeref}" type="{$scs}{@typeref}" minOccurs="0" maxOccurs="1"/>
</xsl:template>

<xsl:template match="scs:element">
    <xs:element name="{@name}" type="{$scs}{@typeref}" minOccurs="0" maxOccurs="unbounded"/>
</xsl:template>

<xsl:template match="scs:attribute" mode="cdata">
    <xs:simpleContent>
        <xs:extension>
            <xsl:attribute name="base">
                <xsl:choose>
                    <xsl:when test="@typeref">
                        <xsl:value-of select="$scs"/>
                        <xsl:value-of select="@typeref"/>
                    </xsl:when>
                    <xsl:otherwise>
                        <xsl:value-of select="@xsdtype"/>
                    </xsl:otherwise>
                </xsl:choose>
            </xsl:attribute>
            <xsl:apply-templates select="../scs:attribute[@xmltype = 'attribute']"/>
        </xs:extension>
    </xs:simpleContent>
</xsl:template>

<xsl:template match="scs:attribute" mode="element">
    <xs:element name="{@name}">
        <xsl:attribute name="type">
            <xsl:choose>
                <xsl:when test="@typeref">
                    <xsl:value-of select="$scs"/>
                    <xsl:value-of select="@typeref"/>
                </xsl:when>
                <xsl:otherwise>
                    <xsl:value-of select="@xsdtype"/>
                </xsl:otherwise>
            </xsl:choose>
        </xsl:attribute>
        <xsl:attribute name="minOccurs">
            <xsl:choose>
                <xsl:when test="@optional = 'true'">0</xsl:when>
                <xsl:otherwise>1</xsl:otherwise>
            </xsl:choose>
        </xsl:attribute>
        <xsl:attribute name="maxOccurs">1</xsl:attribute>
    </xs:element>
</xsl:template>

<xsl:template match="scs:attribute">
    <xs:attribute name="{@name}">
        <xsl:attribute name="type">
            <xsl:choose>
                <xsl:when test="@typeref">
                    <xsl:value-of select="$scs"/>
                    <xsl:value-of select="@typeref"/>
                </xsl:when>
                <xsl:otherwise>
                    <xsl:value-of select="@xsdtype"/>
                </xsl:otherwise>
            </xsl:choose>
        </xsl:attribute>
        <xsl:if test="@optional = 'false'">
            <xsl:attribute name="use">required</xsl:attribute>
        </xsl:if>
    </xs:attribute>
</xsl:template>

</xsl:transform>

