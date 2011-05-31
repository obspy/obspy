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
      logical log,prnt(3),truth,yes_km
      character*8 phcd(max),phlst(10)
C
C       changed modnam to 80 max
c
      character*500 modnam
      dimension tt(max),toang(max),dtdd(max),dtdh(max),dddp(max)
        CHARACTER*500 getstring
clocal  character*40 arcdatdir,arcdat*40/'ARCDAT'/
      dimension usrc(2)
c      data in/1/,modnam/'iasp91'/,phlst(1)/'query'/,prnt(3)/.true./
      data in/1/,phlst(1)/'query'/,prnt(3)/.true./,tokm/111.19/
      data rzero/6371.0/



C     The distance in degree.
C     deltain = 10.0
C     The source depth in km.
C     zs = 10.0
C     The model location and name.
C     modnam = '/Users/lion/temp/iaspei-tau/tables/iasp91'

c
        rd = 45.0/atan(1.0)
c     call assign(2,2,'ttimes.lst')
      prnt(1) = .false.
      prnt(2) = .false.
      log = .true.
C        modnam = getstring('v-z model path')
C        write(*,*) modnam
clocal          string = getstring('v-z model')
clocal          call getenv(ARCDAT,arcdatdir)
clocal          if (arcdatdir(1:1) .ne. '/') then
clocal            modnam = string
clocal          else
clocal            modnam =
clocal     1        arcdatdir(1:lenc(arcdatdir))//'vz_mods/'//string
clocal          end if
cjas      modnam = ' '
      call tabin(in,modnam)
      call brnset(1,phlst,prnt)
c                                    choose source depth
        yes_km = .false.
C       write(*,*) 'Epicent. distances may be in deg. or km'
C       yes_km = truth('Will you use kilometers? (Y or N) ..[N]')
C       if (yes_km) write (2,*) 'Distances are in kilometers'
C     if (log) then
C       write(*,*) 'Pprompts for focal depth & epicentral distance'
C       write(*,*) 'A negative depth terminates the program'
C       write(*,*) 'A negative distance => prompt for new depth'
C       write(*,*)
C     end if
C3    call query('Source depth (km):',log)
C     read(*,*)zs

C     write(*,*) 'Source depth (km):'
C     write(*,*) zs
      if(zs.lt.0.) go to 13
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
C     write(2,*)
C     if (log) then
C       write(*,*)
C       write(*,*) 'You will have to enter a distance,'
C       write(*,*) 'if this is negative a new depth is calculated'
C       write(*,*) 'TO EXIT: give negative depth'
C       write(*,*)
C     end if
c     Write(2,'(a,f6.2)') 'Source Depth: ',zs
c     write(*,*)
c     Write(*,'(a,f6.2)') 'Source Depth: ',zs
Cjas    write(*,*) 'P and S velocities at source depth:',vpfoc,vsfoc
Cjas    write(2,*) 'P and S velocities at source depth:',vpfoc,vsfoc
c                                    loop on delta
C1    write(*,*)
C     write(2,*)
C     call query('Enter delta:',log)
C     read(*,*)deltain
C     if(deltain.lt.0.) go to 3
        if (yes_km) then
          delta = deltain/tokm
        else
          delta = deltain
        end if
      call trtm(delta,max,n,tt,dtdd,dtdh,dddp,phcd)
C     if(n.le.0) go to 2
C     write(*,*)
C    %' delta   #   code       time   Take-off ang   dT/dD',
C    %'    dT/dh     d2T/dD2'
C     write(2,*)
C    %' delta   #   code       time   Take-off ang   dT/dD',
C    %'    dT/dh     d2T/dD2'
        do j=1,n
                if (phcd(j)(1:1).eq.'P' .or. phcd(j)(1:1).eq.'p') then
                        toang(j) = rd*asin(abs(dtdd(j))/etafocp)
                else
                        toang(j) = rd*asin(abs(dtdd(j))/etafocs)
                end if
                if (dtdh(j) .gt. 0.0) toang(j) = 180.-toang(j)
        end do
c
C     write(*,100)deltain,(i,phcd(i),tt(i),toang(i),dtdd(i),dtdh(i),
C    1 dddp(i),i=1,n)
C     write(2,100)deltain,(i,phcd(i),tt(i),toang(i),dtdd(i),dtdh(i),
C    1 dddp(i),i=1,n)
 100  format(1x,f7.3,i3,2x,a,f10.3,f11.3,f10.3,1p2e11.2/
     1 (7x,i4,2x,a,0pf10.3,f11.3,f10.3,1p2e11.2))
C     go to 1
C2    write(*,101)deltain
C      write(2,101)deltain
C 101  format(/1x,'No arrivals for delta =',f7.2)
C      go to 1
c                                    end delta loop
 13   end
C13   call retrns(in)
c      call retrns(10)
C     call exit(0)
C+
        subroutine cstring(string,nstring)
C
C       Input a character string with a read(*,'(A)') string
C       If first two characters are /* it will read the next entry
C       Tab is a delimiter.
C       Returns string and nstring, number of characters to tab.
C       string stars with first non-blank character.
C-
        logical more
        CHARACTER*1 TAB
c       parameter (tab=char(9))
        CHARACTER*(*) string
C
        tab = char(9)
        more = .true.
        do while (more)
          read(*,'(A)') string
          nstring = lenc(string)
          more = (nstring.ge.2 .and. string(1:2).eq.'/*')
        end do
        IF (nstring .GT. 0) THEN
          NTAB = INDEX(string(1:nstring),TAB)
          IF (NTAB .GT. 0) nstring = NTAB - 1
        end if
        return
        end
c+
        character*(*) function GETSTRING(prompt)
c
c  outputs 'prompt' using PRINTX
c  and accepts input character string
c                               Alan Linde ... Aug 1986
C       27 July 1993: Did input read through cstring so can have 
C         comment lines
C       12 February 95:  Kill leading blanks
C       29 August 1997: changed ,getstring=' ' to same with no ,
c-
        character*(*) prompt
        character*80 temp
        getstring = ' '
c output 'prompt'
        call printx(prompt)
        kk=lenc(prompt)
        if (prompt(kk:kk).eq.']') then
          ll=0
          do i=kk-1,1,-1
            if (prompt(i:i).eq.'['.and.ll.eq.0) ll=i+1
          end do
          if (ll.ne.0) getstring=prompt(ll:kk-1)
        end if
c  get the response
        call cstring(temp,nout)
c  Kill leading blanks
        do while (nout.gt.1 .and. temp(1:1).eq.' ')
          nout = nout - 1
          temp(1:nout) = temp(2:nout+1)
          temp(nout+1:nout+1) = ' '
        end do
        if (nout .gt. 0) getstring=temp(1:nout)
        return
        end
C+
        SUBROUTINE IYESNO(MSG,IANS)
C
C
C     PURPOSE:
C            THIS LITTLE SUBROUTINE ASKS A QUESTION AND RETURNS A
C            RESPONSE TO THAT QUESTION. THE ANSWER TO THE QUESTION
C            MUST BE EITHER 'Y' FOR YES, 'N' FOR NO, OR NOTHING
C            (i.e. simply hitting carrage return) FOR THE DEFAULT
C            REPONSE TO THE QUESTION.
C
C     ON INPUT:
C           MSG = BYTE STRING CONTAINING THE QUESTION
C
C     ON OUTPUT:
C           IANS = THE LOGICAL REPONSE TO THE QUESTION (1 or 0)
C     EXTRA FEATURES:
C           DEFAULT SITUATION IS:
C           IF LAST 3 CHARACTERS IN 'MSG' ARE
C                    [Y]  OR  [N]
C           THEN 'IANS' = 1   OR   0
C
C           IF LAST 3 CHARACTERS ARE NOT ONE OF ABOVE PAIRS
C           THEN 'IANS' = 0
C           (i.e. default for no supplied default is N)
C       30 JULY 1989:  IF ENTERED CHARACTER IS A BLANK OR A TAB, 
C           TREATS AS A NULL ENTRY.
C       27 July 1993: Did input read through cstring so can have 
C         comment lines
C-
        CHARACTER*1 DELIM/'$'/,CHARIN,BLANK/' '/
        CHARACTER*3 TEST,UCY,LCY
        character*80 string_in
        CHARACTER*(*) MSG
        DATA UCY/'[Y]'/,LCY/'[y]'/
        KK = LEN(MSG)
        IF (MSG(KK:KK) .EQ. DELIM) KK = KK - 1
        TEST = MSG(KK-2:KK)
        CALL PRINTX(MSG)
        call cstring(string_in,nchar)
        IF ((NCHAR.GT.0) .AND. (string_in(1:1).EQ.BLANK))
     1    NCHAR = 0
        IF (NCHAR .EQ. 0) THEN
          IF ((TEST .EQ. UCY) .OR. (TEST .EQ. LCY)) THEN
            IANS = 1
          ELSE 
            IANS = 0
          END IF
        ELSE
          charin = string_in(1:1)
          IF (CHARIN .EQ. UCY(2:2) .OR. CHARIN .EQ. LCY(2:2)) THEN
            IANS = 1
          ELSE
            IANS = 0
          END IF
        END IF
        RETURN
        END
        SUBROUTINE PRINTX(LINE)
C+
c       SUBROUTINE PRINTX(LINE)
C  OUTPUTS A MESSAGE TO THE TERMINAL
C  PRINTX STARTS WITH A LINE FEED BUT DOES NOT END WITH A CARRIAGE RETURN
C  THE PRINT HEAD REMAINS AT THE END OF THE MESSAGE
C
C  IF THE MESSAGE LENGTH IS LESS THAN 40,
C       DOTS ARE INSERTED UP TO COL. 39
C       AND A COLON IS PUT IN COL. 40.
C
C  USE FOR CONVERSATIONAL INTERACTION
C                       Alan Linde ... April 1980.
C       10 Sugust 1985:  Corrected a minor error for  strings > 40 bytes
C       20 June 1986:  Made it compatible with Fortran 77
C-
        character*(*) line
        CHARACTER*60 BUF
        CHARACTER*2 COLON
        CHARACTER*1 DOT,DELIM
        DATA DELIM/'$'/,DOT/'.'/,COLON/': '/
        KK = lenc(LINE) !  length minus right-hand blanks
          IF (LINE(KK:KK) .EQ. DELIM) KK = KK - 1
          IF (KK .GT. 58) KK = 59
        BUF(1:KK) = LINE(1:KK)
        IF (KK .LT. 49) THEN
          DO J=KK+1,49
            BUF(J:J) = DOT
          END DO
          KK = 49
        END IF
        BUF(KK:KK+1) = COLON
        KK = KK + 1
c       Add a 1x so does not write in column 1 (problem on some platforms)
        WRITE(*,'(1x,A,$)') BUF(1:KK)
        RETURN
        END
C+
        LOGICAL FUNCTION TRUTH(MSG)
C
C
C PURPOSE:
C               ROUTINE ACCEPTS A MESSAGE (QUESTION) REQUIRING A Y/N RESPONSE
C               AND THE RETURN "TRUTH" IS SET:
C                       TRUTH = .TRUE.    IF RESPONSE IS Y(y)
C                       TRUTH = .FALSE.                 N(n)
C
C ROUTINES CALLED:
C                       IYESNO
C                               WHICH CALLS
C                                               NSTRNG
C                                               PRINTX
C
C
C USE:
C       I=TRUTH('ANSWER Y OR N')
C       IF (I) ......
C
C OR
C       IF (TRUTH('REPLY Y OR N')) ....
C
C
C AUTHOR:                       ALAN LINDE ... AUGUST 1980
C
C  ENTRY
C                       ILOGIC (Alan's original name)
C-
        CHARACTER*(*) MSG
        LOGICAL ILOGIC
        ENTRY ILOGIC(MSG)
        TRUTH=.FALSE.
        CALL IYESNO(MSG,IANS)
        IF (IANS.EQ.1) TRUTH=.TRUE.
        ILOGIC = TRUTH
        RETURN
        END
