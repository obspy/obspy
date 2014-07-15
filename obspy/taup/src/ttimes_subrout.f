      subroutine ttimes(deltain, zs, modnam, phcd, tt, toang, dtdd,
     1                  dtdh, dddp)
C      jas version of kennett/buland program
c       July 2008: most recent modifications
C
C
C
C     May 2011: Modified so that it can be accessed from the outside by
C     the obspy dev team: devs@obspy.org
C
C
C
c-
      save
      parameter (max=60)
      logical prnt(3),yes_km
      character*8 phcd(max),phlst(10)
C
C       changed modnam to 80 max
c
      character*500 modnam
      dimension tt(max),toang(max),dtdd(max),dtdh(max),dddp(max)
      dimension usrc(2)
      data in/1/,phlst(1)/'query'/,prnt(3)/.true./,tokm/111.19/
      data rzero/6371.0/

C     The distance in degree.
C     deltain = 10.0
C     The source depth in km.
C     zs = 10.0
C     The model location and name.
C     modnam = '/Users/lion/temp/iaspei-tau/tables/iasp91'

      rd = 45.0/atan(1.0)
      prnt(1) = .false.
      prnt(2) = .false.
      yes_km = .false.
      call tabin(in,modnam)
      call brnset(1,phlst,prnt)

      if (zs.ge.0.) then
        call depset(zs,usrc)
C
C       usrc returns the P and S slownesses -- transformed to flat-earth
C       (28 Sept) If only P or S, may get returned as zero 
C
        etafocp = 0.0
        etafocs = 0.0
        if (usrc(1) .gt. 0.0) then
          vpfoc = ((rzero-zs)/rzero)/usrc(1)
          etafocp = (rzero-zs)/(vpfoc*rd)
        end if
        if (usrc(2) .gt. 0.0) then
          vsfoc = ((rzero-zs)/rzero)/usrc(2)
          etafocs = (rzero-zs)/(vsfoc*rd)
        end if
        if (yes_km) then
          delta = deltain/tokm
        else
          delta = deltain
        end if
        call trtm(delta,max,n,tt,dtdd,dtdh,dddp,phcd)
        do j=1,n
                if (phcd(j)(1:1).eq.'P' .or. phcd(j)(1:1).eq.'p') then
                        toang(j) = rd*asin(abs(dtdd(j))/etafocp)
                else
                        toang(j) = rd*asin(abs(dtdd(j))/etafocs)
                end if
                if (dtdh(j) .gt. 0.0) toang(j) = 180.-toang(j)
        end do
      endif
      end
