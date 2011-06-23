      program evtest
c This program calls the instrument response package and writes out 
c a file suitable for plotting
c
      call wresp
      stop
      end
c
c---------------------------------------------
c
      subroutine wresp
c
c writes out a single instrument response
c
      integer evresp
      character file*80, datime*20
      character*5  unts,sta,cha,net,locid,rtyp
      character*10 vbs
      integer npts, start_stage, stop_stage, stdio_flag
      dimension freq(10000),resp(20000)
      data pi/3.14159265358979/

c set start_stage to -1, stop_stage to 0 (this will cause all of the
c response stages to be included in the response that is calculated)

      start_stage = -1
      stop_stage = 0

c set the stdio_flag to zero (the response is returned to the calling routine

      stdio_flag = 0

c
c hardwire the values of strings
c *** may need to set up new strings, e.g. "IU", etc. ******
c

      sta = 'CTAO'
      cha = 'BHZ'
      net = '*'
      locid = '*'
      datime = '1995,260'
      unts = 'VEL'
c      file = '/users/tjm/eg_responses'
c if above line commented out and following line not, must
c define the SEEDRESP environment variable
      file = 'aa'
      vbs = '-v'
      npts = 100
      rtyp = 'CS'
c
c set up 100 frequency points
c
      fhigh = 100.
      flow = .0001
      fpts = 100.
      df = (alog10(fhigh) - alog10(flow) ) / (fpts-1)
      
      do 10 i=1,npts
        freq(i) = 10.0**( alog10(flow) + float(i-1) * df )
 10   continue
c
c now call evresp, assume resp() will contain the multiplexed output
c 
        
       iflag = evresp(sta,cha,net,locid,datime,unts,file,freq,npts,resp,
     .                rtyp,vbs,start_stage,stop_stage,stdio_flag)
       if (iflag.ne.0) then
          write(*,31) 'ERROR processing: ',sta,cha,datime
 31       format(a,1x,a5,1x,a5,1x,a20)
          return
       endif
c 
c  write out file of frequency, amplitude, phase
c     
      open(8,file='evtest.out')
      j = 1
      do 30 i=1,fpts
        amp = sqrt(resp(j)**2 + resp(j+1)**2)
        pha = atan2(resp(j+1),resp(j)) * 180. / pi
        write(8,'(3e15.6)') freq(i),amp,pha
        j = j + 2
 30   continue
      close(8) 

      return
      end
