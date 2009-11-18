"""Convenience functions for handling SAC files\n
written by C. J. Ammon, Penn State"""

class PyTutil:


    def __init__(self):
        pass


    def is_leap_year(self,year):
        """ returns 1 if year is leap year; 0 otherwise"""
	isleap = 0
	i0 = year % 4
	i1 = year % 100
	i2 = year % 400
	if ((i0 == 0 and i1 != 0) or i2 == 0):
	  isleap = 1
	#
	return(isleap)

    def ndaysinyear(self,year):
        """ returns number of days in year\n"""
	if (self.is_leap_year(year) == 1):
            return(366)
	return(365)



    def doy(self,y,m,d):
        """return year day for year (y), month (m) and day of month (d)\n """
        	#
	day = int(d)
	#
	if m == 1:
            return(day)
	#
	nd = [31,28,31,30,31,30,31,31,30,31,30,31]
	#
	if (self.is_leap_year(y) == 1):
            nd[1] = 29
            #
        for i in range(int(m) - 1):
            day = day + nd[i]
            #
            #
        return(day)


    def monthdate(self, year,day):
        """ return day of month for given year and yearday"""
	ok = -1
	if(day > self.ndaysinyear(year)):
            return([0,0,ok])
	#
	nd = [31,28,31,30,31,30,31,31,30,31,30,31]
	if (self.is_leap_year(year) == 1):
            nd[1] = 29
	d = day
	#
	m = 1
	ok = 1
	for i in range(12):
            if ( d <= nd[i]):
                date = d
                return([m, date, ok])
            m = m + 1
            d = d - nd[i]
	ok = -1
	return([0,0,ok])


    def yd2seconds(y,d):
        """ calculate number of seconds since 1970 for \n
        year (y) and yearday (d)"""
	ok = 1
	y0 = 1970
	y1 = y - y0
	ndays = 0
	for iy in range(y0,y):
		ndays = ndays + self.ndaysinyear(iy)
	ndays = ndays + d
	nseconds = ndays * 24 * 60 * 60
	return([nseconds, ok])


    def dt2seconds(y,d,h,m,s):
        """ calculate number of seconds since 1970 for \n
        year (y), yearday (d), hour (h), ,minute (m) and second (s)"""
	ok = 1
	nseconds = self.yd2seconds(y,d)[0] + h*60*60 + m*60 + s
	return([nseconds, ok])

