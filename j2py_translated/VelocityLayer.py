#!/usr/bin/env python
""" generated source for module VelocityLayer """
# 
#  * The TauP Toolkit: Flexible Seismic Travel-Time and Raypath Utilities.
#  * Copyright (C) 1998-2000 University of South Carolina
#  * 
#  * This program is free software; you can redistribute it and/or modify it under
#  * the terms of the GNU General Public License as published by the Free Software
#  * Foundation; either version 2 of the License, or (at your option) any later
#  * version.
#  * 
#  * This program is distributed in the hope that it will be useful, but WITHOUT
#  * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
#  * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
#  * details.
#  * 
#  * You should have received a copy of the GNU General Public License along with
#  * this program; if not, write to the Free Software Foundation, Inc., 59 Temple
#  * Place - Suite 330, Boston, MA 02111-1307, USA.
#  * 
#  * The current version can be found at <A
#  * HREF="www.seis.sc.edu">http://www.seis.sc.edu</A>
#  * 
#  * Bug reports and comments should be directed to H. Philip Crotwell,
#  * crotwell@seis.sc.edu or Tom Owens, owens@seis.sc.edu
#  * 
#  
# 
#  * package for storage and manipulation of seismic earth models.
#  * 
#  
# package: edu.sc.seis.TauP
import java.io.Serializable

# 
#  * The VelocityModelLayer class stores and manipulates a singly layer. An entire
#  * velocity model is implemented as an Vector of layers.
#  * 
#  * @version 1.1.3 Wed Jul 18 15:00:35 GMT 2001
#  * 
#  * 
#  * 
#  * @author H. Philip Crotwell
#  
class VelocityLayer(Cloneable, Serializable):
    """ generated source for class VelocityLayer """
    myLayerNumber = int()
    topDepth = float()
    botDepth = float()
    topPVelocity = float()
    botPVelocity = float()
    topSVelocity = float()
    botSVelocity = float()
    topDensity = 2.6
    botDensity = 2.6
    topQp = 1000
    botQp = 1000
    topQs = 2000
    botQs = 2000

    @overloaded
    def __init__(self, myLayerNumber, topDepth, botDepth, topPVelocity, botPVelocity, topSVelocity, botSVelocity):
        """ generated source for method __init__ """
        super(VelocityLayer, self).__init__()
        self.__init__(myLayerNumber, topDepth, botDepth, topPVelocity, botPVelocity, topSVelocity, botSVelocity, 2.6, 2.6)

    @__init__.register(object, int, float, float, float, float, float, float, float, float)
    def __init___0(self, myLayerNumber, topDepth, botDepth, topPVelocity, botPVelocity, topSVelocity, botSVelocity, topDensity, bottomDensity):
        """ generated source for method __init___0 """
        super(VelocityLayer, self).__init__()
        self.__init__(myLayerNumber, topDepth, botDepth, topPVelocity, botPVelocity, topSVelocity, botSVelocity, topDensity, bottomDensity, 1000, 1000, 2000, 2000)

    @__init__.register(object, int, float, float, float, float, float, float, float, float, float, float, float, float)
    def __init___1(self, myLayerNumber, topDepth, botDepth, topPVelocity, botPVelocity, topSVelocity, botSVelocity, topDensity, botDensity, topQp, botQp, topQs, botQs):
        """ generated source for method __init___1 """
        super(VelocityLayer, self).__init__()
        if topPVelocity <= 0:
            raise IllegalArgumentException("topPVelocity must be positive: " + topPVelocity)
        if botPVelocity <= 0:
            raise IllegalArgumentException("botPVelocity must be positive: " + botPVelocity)
        if topSVelocity < 0:
            raise IllegalArgumentException("topSVelocity must be nonnegative: " + topSVelocity)
        if botSVelocity < 0:
            raise IllegalArgumentException("botSVelocity must be nonnegative: " + botSVelocity)
        self.myLayerNumber = myLayerNumber
        self.topDepth = topDepth
        self.botDepth = botDepth
        self.topPVelocity = topPVelocity
        self.botPVelocity = botPVelocity
        self.topSVelocity = topSVelocity
        self.botSVelocity = botSVelocity
        self.topDensity = topDensity
        self.botDensity = botDensity
        self.topQp = topQp
        self.botQp = botQp
        self.topQs = topQs
        self.botQs = botQs

    def clone(self):
        """ generated source for method clone """
        try:
            return newObject
        except CloneNotSupportedException as e:
            #  Cannot happen, we support clone
            #  and our parent is Object, which supports clone.
            raise InternalError(e.__str__())

    def evaluateAtBottom(self, materialProperty):
        """ generated source for method evaluateAtBottom """
        answer = float()
        if materialProperty=='P':
            pass
        elif materialProperty=='p':
            answer = getBotPVelocity()
        elif materialProperty=='s':
            pass
        elif materialProperty=='S':
            answer = getBotSVelocity()
        elif materialProperty=='r':
            pass
        elif materialProperty=='R':
            pass
        elif materialProperty=='D':
            pass
        elif materialProperty=='d':
            answer = getBotDensity()
        else:
            raise NoSuchMatPropException(materialProperty)
        return answer

    def evaluateAtTop(self, materialProperty):
        """ generated source for method evaluateAtTop """
        answer = float()
        if materialProperty=='P':
            pass
        elif materialProperty=='p':
            answer = getTopPVelocity()
        elif materialProperty=='s':
            pass
        elif materialProperty=='S':
            answer = getTopSVelocity()
        elif materialProperty=='r':
            pass
        elif materialProperty=='R':
            pass
        elif materialProperty=='D':
            pass
        elif materialProperty=='d':
            answer = getTopDensity()
        else:
            raise NoSuchMatPropException(materialProperty)
        return answer

    def evaluateAt(self, depth, materialProperty):
        """ generated source for method evaluateAt """
        slope = float()
        answer = float()
        if materialProperty=='P':
            pass
        elif materialProperty=='p':
            slope = (getBotPVelocity() - getTopPVelocity()) / (getBotDepth() - getTopDepth())
            answer = slope * (depth - getTopDepth()) + getTopPVelocity()
        elif materialProperty=='s':
            pass
        elif materialProperty=='S':
            slope = (getBotSVelocity() - getTopSVelocity()) / (getBotDepth() - getTopDepth())
            answer = slope * (depth - getTopDepth()) + getTopSVelocity()
        elif materialProperty=='r':
            pass
        elif materialProperty=='R':
            pass
        elif materialProperty=='D':
            pass
        elif materialProperty=='d':
            slope = (getBotDensity() - getTopDensity()) / (getBotDepth() - getTopDepth())
            answer = slope * (depth - getTopDepth()) + getTopDensity()
        else:
            print "I don't understand this material property: " + materialProperty + "\nUse one of P p S s R r D d"
            raise NoSuchMatPropException(materialProperty)
        return answer

    def __str__(self):
        """ generated source for method toString """
        description = str()
        description = self.myLayerNumber + " " + getTopDepth() + " " + getBotDepth()
        description += " P " + getTopPVelocity() + " " + getBotPVelocity()
        description += " S " + getTopSVelocity() + " " + getBotSVelocity()
        description += " Density " + getTopDensity() + " " + getBotDensity()
        return description

    def getLayerNum(self):
        """ generated source for method getLayerNum """
        return self.myLayerNumber

    def setTopDepth(self, topDepth):
        """ generated source for method setTopDepth """
        self.topDepth = topDepth

    def getTopDepth(self):
        """ generated source for method getTopDepth """
        return self.topDepth

    def setBotDepth(self, botDepth):
        """ generated source for method setBotDepth """
        self.botDepth = botDepth

    def getBotDepth(self):
        """ generated source for method getBotDepth """
        return self.botDepth

    def setTopPVelocity(self, topPVelocity):
        """ generated source for method setTopPVelocity """
        self.topPVelocity = topPVelocity

    def getTopPVelocity(self):
        """ generated source for method getTopPVelocity """
        return self.topPVelocity

    def setBotPVelocity(self, botPVelocity):
        """ generated source for method setBotPVelocity """
        self.botPVelocity = botPVelocity

    def getBotPVelocity(self):
        """ generated source for method getBotPVelocity """
        return self.botPVelocity

    def setTopSVelocity(self, topSVelocity):
        """ generated source for method setTopSVelocity """
        self.topSVelocity = topSVelocity

    def getTopSVelocity(self):
        """ generated source for method getTopSVelocity """
        return self.topSVelocity

    def setBotSVelocity(self, botSVelocity):
        """ generated source for method setBotSVelocity """
        self.botSVelocity = botSVelocity

    def getBotSVelocity(self):
        """ generated source for method getBotSVelocity """
        return self.botSVelocity

    def setTopDensity(self, topDensity):
        """ generated source for method setTopDensity """
        self.topDensity = topDensity

    def getTopDensity(self):
        """ generated source for method getTopDensity """
        return self.topDensity

    def setBotDensity(self, botDensity):
        """ generated source for method setBotDensity """
        self.botDensity = botDensity

    def getBotDensity(self):
        """ generated source for method getBotDensity """
        return self.botDensity

    def setTopQp(self, topQp):
        """ generated source for method setTopQp """
        self.topQp = topQp

    def getTopQp(self):
        """ generated source for method getTopQp """
        return self.topQp

    def setBotQp(self, botQp):
        """ generated source for method setBotQp """
        self.botQp = botQp

    def getBotQp(self):
        """ generated source for method getBotQp """
        return self.botQp

    def setTopQs(self, topQs):
        """ generated source for method setTopQs """
        self.topQs = topQs

    def getTopQs(self):
        """ generated source for method getTopQs """
        return self.topQs

    def setBotQs(self, botQs):
        """ generated source for method setBotQs """
        self.botQs = botQs

    def getBotQs(self):
        """ generated source for method getBotQs """
        return self.botQs

    def getThickness(self):
        """ generated source for method getThickness """
        return self.getBotDepth() - self.getTopDepth()

