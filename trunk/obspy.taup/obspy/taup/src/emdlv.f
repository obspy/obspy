C  emdlv(r,vp,vs)
C+
      subroutine emdlv(r,vp,vs)
C
C  Replaces the routine with the same name and its entry, emdld, in the
C    IASPEI91 package.  This version allows introduction of other models
C    to prodouce tables.  REMODL calls emdld first.  Here it opens a file
C    containing the model (ascii).  The first two lines are comment lines
C    for the P and S models respectively.  The remaining lines are z, vp,
C    vs, density per line in free format starting from the surface down.
C    (Density is not used and could be left out.)  First order discontinuities
C    are included by repeated z values.  At this stage the user is prompted
C    for the model name and the filespec.  The calls to emdlv use linear
C    interpolation to get the vp and vs for the desired r.  R0=6371.0 is
C    assumed.  Limit is 200 layers and a maximum of 30 discontinuities.
C
C  Arthur Snoke  VTSO  5 April 1991.  Modified some by Brian Kennett ANU
C-
      save
      parameter (max=200, npmax=30)
      logical ldep
      character*(*) name
      character*8 arg1,tvelnam
      character*80 filespec, dummy
      real*4 zin(max), vpin(max), vsin(max), rd(npmax)
c
      depth = AMAX1(6371.0 - r,0.0)
      ldep = .false.
      i = 1
      do while (.not.ldep .and. i .le.nz)
        if (zin(i) .le. depth) then
          if (zin(i) .eq. depth) then
            vp = vpin(i)
            vs = vsin(i)
            return
          else
            i = i + 1
          end if
        else
          ldep = .true.
        end if
      end do
      if (ldep) then
        vp=vpin(i-1)+(vpin(i)-vpin(i-1))
     *    *(depth-zin(i-1))/(zin(i)-zin(i-1))
        vs=vsin(i-1)+(vsin(i)-vsin(i-1))
     *    *(depth-zin(i-1))/(zin(i)-zin(i-1))
        return
      else
        vp = vpin(nz)
        vs = vsin(nz)
        return
      end if
C
      entry emdld(np,rd,name)
c      call query('Model name..[return for IASP91]..:',log)
c      read(*,'(a)',iostat=ierr) name
c      if (ierr .ne. 0) name = 'iasp91'
c      call query('V-Z model filespec..:',LOG)
c      read(*,'(A)') filespec
c                                      input from command line
       call getarg(1,arg1)
       read(arg1,*) tvelnam
       name = tvelnam
       filespec = tvelnam(1:lenc(tvelnam))//".tvel"
c
      close(unit=13)
      call assign(13,1,filespec)
      read(13,'(a)') dummy
      read(13,'(a)') dummy
c
      nin=1
      ierr = 0
      do while ((ierr .eq. 0) .and. (nin.le.max))
c
c  read in velocity-depth models (both p and s)
c
        read(13,*,iostat=ierr) zin(nin), vpin(nin), vsin(nin)
        if (ierr .eq. 0) then
          nz = nin
          nin = nin + 1
        end if
      end do
      close(unit=13)
c
c      now for the discontinuities
c
      np = 0
      do j=nz-1,2,-1
        if (zin(j) .eq. zin(j+1)) then
          np = np + 1
          rd(np) = 6371.0 - zin(j)
        end if
      end do
      np = np + 1
      rd(np) = 6371.0
      return
      end
c  end emdlv.for
