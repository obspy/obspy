C+
C	libtau.f is from Buland's 1996 Feb. libtau.src.
C	Added references to my lenc routine for calculating lengths
C	  of character strings up to first blank or null (jas/vt)
C
C
C
C
C     May 2011: Modified so that it can be accessed from the outside by
C     the obspy dev team: devs@obspy.org
C
C
C
C
C-
      subroutine tabin(in,modnam)
      include 'ttlim.inc'
      character*(*) modnam
      character*500 filename
c     logical log
      character*8 phcd,phdif(9)
      double precision pm,zm,us,pt,tau,xlim,xbrn,dbrn,zs,pk,pu,pux,tauu,
     1 xu,px,xt,taut,coef,tauc,xc,tcoef,tp
c
      common/umdc/pm(jsrc,2),zm(jsrc,2),ndex(jsrc,2),mt(2)
      common/tabc/us(2),pt(jout),tau(4,jout),xlim(2,jout),xbrn(jbrn,3),
     1 dbrn(jbrn,2),xn,pn,tn,dn,hn,jndx(jbrn,2),idel(jbrn,3),mbr1,mbr2
      common/brkc/zs,pk(jseg),pu(jtsm0,2),pux(jxsm,2),tauu(jtsm,2),
     1 xu(jxsm,2),px(jbrn,2),xt(jbrn,2),taut(jout),coef(5,jout),
     2 tauc(jtsm),xc(jxsm),tcoef(5,jbrna,2),tp(jbrnu,2),odep,
     3 fcs(jseg,3),nin,nph0,int0(2),ki,msrc(2),isrc(2),nseg,nbrn,ku(2),
     4 km(2),nafl(jseg,3),indx(jseg,2),kndx(jseg,2),iidx(jseg),
     5 jidx(jbrn),kk(jseg)
      common/pcdc/phcd(jbrn)
      data tauc,xc/jtsm*0d0,jxsm*0d0/
      tauu = 0d0
      xu = 0d0

c
      nin=in
      phdif(1)='P'
      phdif(2)='S'
      phdif(3)='pP'
      phdif(4)='sP'
      phdif(5)='pS'
      phdif(6)='sS'
      phdif(7)='PKPab'
      phdif(8)='pPKPab'
      phdif(9)='sPKPab'
c++
      call asnag1(nin,-1,1,'Enter model name:',modnam)
c++ 
      read(nin) nasgr,nl,len2,xn,pn,tn,mt,nseg,nbrn,ku,km,fcs,nafl,
     1 indx,kndx
      read(nin) pm,zm,ndex
      read(nin) pu,pux
      read(nin) phcd,px,xt,jndx
      read(nin) pt,taut
      read(nin) coef
      call retrns(nin)
c
	nb = lenc(modnam)
      filename = modnam(1:nb)//'.tbl'
      call dasign(nin,-1,filename,nasgr)
c
      do 11 nph=1,2
 11   pu(ku(nph)+1,nph)=pm(1,nph)
c
c     write(10,*)'nasgr nl len2',nasgr,nl,len2
c     write(10,*)'nseg nbrn mt ku km',nseg,nbrn,mt,ku,km
c     write(10,*)'xn pn tn',xn,pn,tn
c     write(10,200)(i,(ndex(i,j),pm(i,j),zm(i,j),j=1,2),i=1,mt(2))
c200  format(/(1x,i3,i7,2f12.6,i7,2f12.6))
c     write(10,201)(i,(pu(i,j),j=1,2),i=1,ku(2)+1)
c201  format(/(1x,i3,2f12.6))
c     write(10,201)(i,(pux(i,j),j=1,2),i=1,km(2))
c     write(10,202)(i,(nafl(i,j),j=1,3),(indx(i,j),j=1,2),(kndx(i,j),
c    1 j=1,2),(fcs(i,j),j=1,3),i=1,nseg)
c202  format(/(1x,i3,7i5,3f5.0))
c     cn=180./3.1415927
c     write(10,203)(i,(jndx(i,j),j=1,2),(px(i,j),j=1,2),(cn*xt(i,j),
c    1 j=1,2),phcd(i),i=1,nbrn)
c203  format(/(1x,i3,2i5,2f12.6,2f12.2,2x,a))
c     write(10,204)(i,pt(i),taut(i),(coef(j,i),j=1,5),i=1,jout)
c204  format(/(1x,i5,0p2f12.6,1p5d10.2))
c
      tn=1./tn
      dn=3.1415927/(180.*pn*xn)
      odep=-1.
      ki=0
      msrc(1)=0
      msrc(2)=0
      k=1
      do 3 i=1,nbrn
      jidx(i)=jndx(i,2)
      do 4 j=1,2
 4    dbrn(i,j)=-1d0
 8    if(jndx(i,2).le.indx(k,2)) go to 7
      k=k+1
      go to 8
 7    if(nafl(k,2).gt.0) go to 9
      ind=nafl(k,1)
      l=0
      do 10 j=jndx(i,1),jndx(i,2)
      l=l+1
 10   tp(l,ind)=pt(j)
 9    if(nafl(k,1).gt.0.and.(phcd(i)(1:1).eq.'P'.or.
     1 phcd(i)(1:1).eq.'S')) go to 3
      do 5 j=1,9
      if(phcd(i).eq.phdif(j)) go to 6
 5    continue
      go to 3
 6    dbrn(i,1)=1d0
      phdif(j)=' '
 3    continue
c     write(10,205)(i,phcd(i),(dbrn(i,j),j=1,2),jidx(i),i=1,nbrn)
c205  format(/(1x,i5,2x,a,2f8.2,i5))
c     write(10,206)(i,(tp(i,j),j=1,2),i=1,jbrnu)
c206  format(/(1x,i5,2f12.6))
      return
      end
      subroutine depset(dep,usrc)
      save 
      include 'ttlim.inc'
      logical dop,dos,segmsk,prnt
      character*8 phcd
      real usrc(2)
      double precision pm,zm,us,pt,tau,xlim,xbrn,dbrn,zs,pk,pu,pux,tauu,
     1 xu,px,xt,taut,coef,tauc,xc,tcoef,tp
      common/umdc/pm(jsrc,2),zm(jsrc,2),ndex(jsrc,2),mt(2)
      common/tabc/us(2),pt(jout),tau(4,jout),xlim(2,jout),xbrn(jbrn,3),
     1 dbrn(jbrn,2),xn,pn,tn,dn,hn,jndx(jbrn,2),idel(jbrn,3),mbr1,mbr2
      common/brkc/zs,pk(jseg),pu(jtsm0,2),pux(jxsm,2),tauu(jtsm,2),
     1 xu(jxsm,2),px(jbrn,2),xt(jbrn,2),taut(jout),coef(5,jout),
     2 tauc(jtsm),xc(jxsm),tcoef(5,jbrna,2),tp(jbrnu,2),odep,
     3 fcs(jseg,3),nin,nph0,int0(2),ki,msrc(2),isrc(2),nseg,nbrn,ku(2),
     4 km(2),nafl(jseg,3),indx(jseg,2),kndx(jseg,2),iidx(jseg),
     5 jidx(jbrn),kk(jseg)
      common/pcdc/phcd(jbrn)
      common/prtflc/segmsk(jseg),prnt(2)
      data segmsk,prnt/jseg*.true.,2*.false./
c
      if(amax1(dep,.011).ne.odep) go to 1
      dop=.false.
      dos=.false.
      do 2 i=1,nseg
      if(.not.segmsk(i).or.iidx(i).gt.0) go to 2
      if(iabs(nafl(i,1)).le.1) dop=.true.
      if(iabs(nafl(i,1)).ge.2) dos=.true.
 2    continue
      if(.not.dop.and..not.dos) return
      go to 3
c
 1    nph0=0
      int0(1)=0
      int0(2)=0
      mbr1=nbrn+1
      mbr2=0
      dop=.false.
      dos=.false.
      do 4 i=1,nseg
      if(.not.segmsk(i)) go to 4
      if(iabs(nafl(i,1)).le.1) dop=.true.
      if(iabs(nafl(i,1)).ge.2) dos=.true.
 4    continue
      do 5 i=1,nseg
      if(nafl(i,2).gt.0.or.odep.lt.0.) go to 5
      ind=nafl(i,1)
      k=0
      do 15 j=indx(i,1),indx(i,2)
      k=k+1
 15   pt(j)=tp(k,ind)
 5    iidx(i)=-1
      do 6 i=1,nbrn
 6    jndx(i,2)=-1
      if(ki.le.0) go to 7
      do 8 i=1,ki
      j=kk(i)
 8    pt(j)=pk(i)
      ki=0
c   Sample the model at the source depth.
 7    odep=amax1(dep,.011)
      rdep=dep
      if(rdep.lt..011) rdep=0.
      zs=amin1(alog(amax1(1.-rdep*xn,1e-30)),0.)
      hn=1./(pn*(1.-rdep*xn))
      if(prnt(1).or.prnt(2)) write(10,100)dep
 100  format(/1x,'Depth =',f7.2/)
c
 3    if(nph0.gt.1) go to 12
      if(dop) call depcor(1)
      if(dos) call depcor(2)
      go to 14
 12   if(dos) call depcor(2)
      if(dop) call depcor(1)
c
c   Interpolate all tau branches.
c
 14   j=1
      do 9 i=1,nseg
      if(.not.segmsk(i)) go to 9
      nph=iabs(nafl(i,1))
c     print *,'i iidx nph msrc nafl =',i,iidx(i),nph,msrc(nph),nafl(i,1)
      if(iidx(i).gt.0.or.(msrc(nph).le.0.and.nafl(i,1).gt.0)) go to 9
      iidx(i)=1
      if(nafl(i,2).le.0) int=nafl(i,1)
      if(nafl(i,2).gt.0.and.nafl(i,2).eq.iabs(nafl(i,1)))
     1  int=nafl(i,2)+2
      if(nafl(i,2).gt.0.and.nafl(i,2).ne.iabs(nafl(i,1)))
     1  int=iabs(nafl(i,1))+4
      if(nafl(i,2).gt.0.and.nafl(i,2).ne.nafl(i,3)) int=nafl(i,2)+6
 11   if(jndx(j,1).ge.indx(i,1)) go to 10
      j=j+1
      go to 11
 10   idel(j,3)=nafl(i,1)
c     print *,'spfit:  j int =',j,int 
      call spfit(j,int)
      mbr1=min0(mbr1,j)
      mbr2=max0(mbr2,j)
      if(j.ge.nbrn) go to 9
      j=j+1
c     print *,'j jidx indx jndx =',j,jidx(j),indx(i,2),jndx(j,2)
      if(jidx(j).le.indx(i,2).and.jndx(j,2).gt.0) go to 10
 9    continue
c     write(10,*)'mbr1 mbr2',mbr1,mbr2
c     write(10,*)'msrc isrc odep zs us',msrc,isrc,odep,sngl(zs),
c    1 sngl(us(1)),sngl(us(2))
c     write(10,200)ki,(i,iidx(i),kk(i),pk(i),i=1,nseg)
c200  format(/10x,i5/(1x,3i5,f12.6))
      usrc(1)=us(1)/pn
      usrc(2)=us(2)/pn
      return
      end
      subroutine depcor(nph)
      save
      include 'ttlim.inc'
      character*8 phcd
      logical noend,noext,segmsk,prnt
      double precision pm,zm,us,pt,tau,xlim,xbrn,dbrn,zs,pk,pu,pux,tauu,
     1 xu,px,xt,taut,coef,tauc,xc,tcoef,tp,ua,taua
      double precision tup(jrec),umod,zmod,tauus1(2),tauus2(2),xus1(2),
     1 xus2(2),ttau,tx,sgn,umin,dtol,u0,u1,z0,z1,fac,du
      common/umdc/pm(jsrc,2),zm(jsrc,2),ndex(jsrc,2),mt(2)
      common/tabc/us(2),pt(jout),tau(4,jout),xlim(2,jout),xbrn(jbrn,3),
     1 dbrn(jbrn,2),xn,pn,tn,dn,hn,jndx(jbrn,2),idel(jbrn,3),mbr1,mbr2
      common/brkc/zs,pk(jseg),pu(jtsm0,2),pux(jxsm,2),tauu(jtsm,2),
     1 xu(jxsm,2),px(jbrn,2),xt(jbrn,2),taut(jout),coef(5,jout),
     2 tauc(jtsm),xc(jxsm),tcoef(5,jbrna,2),tp(jbrnu,2),odep,
     3 fcs(jseg,3),nin,nph0,int0(2),ki,msrc(2),isrc(2),nseg,nbrn,ku(2),
     4 km(2),nafl(jseg,3),indx(jseg,2),kndx(jseg,2),iidx(jseg),
     5 jidx(jbrn),kk(jseg)
      common/pcdc/phcd(jbrn)
      common/pdec/ua(5,2),taua(5,2),deplim,ka
      common/prtflc/segmsk(jseg),prnt(2)
      equivalence (tauc,tup)
      data tol,dtol,deplim,ka,lpower/.01,1d-6,1.1,4,7/
c
c     write(10,*)'depcor:  nph nph0',nph,nph0
      if(nph.eq.nph0) go to 1
      nph0=nph
      us(nph)=umod(zs,isrc,nph)
c   If we are in a high slowness zone, find the slowness of the lid.
      umin=us(nph)
      ks=isrc(nph)
c     write(10,*)'ks us',ks,sngl(umin)
      do 2 i=1,ks
      if(pm(i,nph).gt.umin) go to 2
      umin=pm(i,nph)
 2    continue
c   Find where the source slowness falls in the ray parameter array.
      n1=ku(nph)+1
      do 3 i=2,n1
      if(pu(i,nph).gt.umin) go to 4
 3    continue
      k2=n1
      if(pu(n1,nph).eq.umin) go to 50
c     call abort('Source slowness too large.')
	write(*,*) 'Source slowness too large.'
	call exit(0) 
 4    k2=i
c50   write(10,*)'k2 umin',k2,sngl(umin)
c
c   Read in the appropriate depth correction values.
c
 50   noext=.false.
      sgn=1d0
      if(msrc(nph).eq.0) msrc(nph)=1
c   See if the source depth coincides with a model sample
      ztol=xn*tol/(1.-xn*odep)
      if(dabs(zs-zm(ks+1,nph)).gt.ztol) go to 5
      ks=ks+1
      go to 6
 5    if(dabs(zs-zm(ks,nph)).gt.ztol) go to 7
c   If so flag the fact and make sure that the right integrals are
c   available.
 6    noext=.true.
      if(msrc(nph).eq.ks) go to 8
      call bkin(nin,ndex(ks,nph),ku(nph)+km(nph),tup)
      go to 11
c   If it is necessary to interpolate, see if appropriate integrals
c   have already been read in.
 7    if(msrc(nph).ne.ks+1) go to 9
      ks=ks+1
      sgn=-1d0
      go to 8
 9    if(msrc(nph).eq.ks) go to 8
c   If not, read in integrals for the model depth nearest the source
c   depth.
      if(dabs(zm(ks,nph)-zs).le.dabs(zm(ks+1,nph)-zs)) go to 10
      ks=ks+1
      sgn=-1d0
 10   call bkin(nin,ndex(ks,nph),ku(nph)+km(nph),tup)
c   Move the depth correction values to a less temporary area.
 11   do 31 i=1,ku(nph)
 31   tauu(i,nph)=tup(i)
      k=ku(nph)
      do 12 i=1,km(nph)
      k=k+1
      xc(i)=tup(k)
 12   xu(i,nph)=tup(k)
c     write(10,*)'bkin',ks,sngl(sgn),sngl(tauu(1,nph)),sngl(xu(1,nph))
c
c   Fiddle pointers.
c
 8    msrc(nph)=ks
c     write(10,*)'msrc sgn',msrc(nph),sngl(sgn)
      noend=.false.
      if(dabs(umin-pu(k2-1,nph)).le.dtol*umin) k2=k2-1
      if(dabs(umin-pu(k2,nph)).le.dtol*umin) noend=.true.
      if(msrc(nph).le.1.and.noext) msrc(nph)=0
      k1=k2-1
      if(noend) k1=k2
c     write(10,*)'noend noext k2 k1',noend,noext,k2,k1
      if(noext) go to 14
c
c   Correct the integrals for the depth interval [zm(msrc),zs].
c
      ms=msrc(nph)
      if(sgn)15,16,16
 16   u0=pm(ms,nph)
      z0=zm(ms,nph)
      u1=us(nph)
      z1=zs
      go to 17
 15   u0=us(nph)
      z0=zs
      u1=pm(ms,nph)
      z1=zm(ms,nph)
 17   mu=1
c     write(10,*)'u0 z0',sngl(u0),sngl(z0)
c     write(10,*)'u1 z1',sngl(u1),sngl(z1)
      do 18 k=1,k1
      call tauint(pu(k,nph),u0,u1,z0,z1,ttau,tx)
      tauc(k)=tauu(k,nph)+sgn*ttau
      if(dabs(pu(k,nph)-pux(mu,nph)).gt.dtol) go to 18
      xc(mu)=xu(mu,nph)+sgn*tx
c     write(10,*)'up x:  k mu',k,mu,sngl(xu(mu,nph)),sngl(xc(mu))
      mu=mu+1
 18   continue
      go to 39
c   If there is no correction, copy the depth corrections to working
c   storage.
 14   mu=1
      do 40 k=1,k1
      tauc(k)=tauu(k,nph)
      if(dabs(pu(k,nph)-pux(mu,nph)).gt.dtol) go to 40
      xc(mu)=xu(mu,nph)
c     write(10,*)'up x:  k mu',k,mu,sngl(xu(mu,nph)),sngl(xc(mu))
      mu=mu+1
 40   continue
c
c   Calculate integrals for the ray bottoming at the source depth.
c
 39   xus1(nph)=0d0
      xus2(nph)=0d0
      mu=mu-1
      if(dabs(umin-us(nph)).gt.dtol.and.dabs(umin-pux(mu,nph)).le.dtol)
     1  mu=mu-1
c   This loop may be skipped only for surface focus as range is not
c   available for all ray parameters.
      if(msrc(nph).le.0) go to 1
      is=isrc(nph)
      tauus2(nph)=0d0
      if(dabs(pux(mu,nph)-umin).gt.dtol.or.dabs(us(nph)-umin).gt.dtol)
     1  go to 48
c   If we happen to be right at a discontinuity, range is available.
      tauus1(nph)=tauc(k1)
      xus1(nph)=xc(mu)
c     write(10,*)'is ks tauus1 xus1',is,ks,sngl(tauus1(nph)),
c    1 sngl(xus1(nph)),'  *'
      go to 33
c   Integrate from the surface to the source.
 48   tauus1(nph)=0d0
      j=1
      if(is.lt.2) go to 42
      do 19 i=2,is
      call tauint(umin,pm(j,nph),pm(i,nph),zm(j,nph),zm(i,nph),ttau,tx)
      tauus1(nph)=tauus1(nph)+ttau
      xus1(nph)=xus1(nph)+tx
 19   j=i
c     write(10,*)'is ks tauus1 xus1',is,ks,sngl(tauus1(nph)),
c    1 sngl(xus1(nph))
 42   if(dabs(zm(is,nph)-zs).le.dtol) go to 33
c   Unless the source is right on a sample slowness, one more partial
c   integral is needed.
      call tauint(umin,pm(is,nph),us(nph),zm(is,nph),zs,ttau,tx)
      tauus1(nph)=tauus1(nph)+ttau
      xus1(nph)=xus1(nph)+tx
c     write(10,*)'is ks tauus1 xus1',is,ks,sngl(tauus1(nph)),
c    1 sngl(xus1(nph))
 33   if(pm(is+1,nph).lt.umin) go to 41
c   If we are in a high slowness zone, we will also need to integrate
c   down to the turning point of the shallowest down-going ray.
      u1=us(nph)
      z1=zs
      do 35 i=is+1,mt(nph)
      u0=u1
      z0=z1
      u1=pm(i,nph)
      z1=zm(i,nph)
      if(u1.lt.umin) go to 36
      call tauint(umin,u0,u1,z0,z1,ttau,tx)
      tauus2(nph)=tauus2(nph)+ttau
 35   xus2(nph)=xus2(nph)+tx
c36   write(10,*)'is ks tauus2 xus2',is,ks,sngl(tauus2(nph)),
c    1 sngl(xus2(nph))
 36   z1=zmod(umin,i-1,nph)
      if(dabs(z0-z1).le.dtol) go to 41
c   Unless the turning point is right on a sample slowness, one more
c   partial integral is needed.
      call tauint(umin,u0,umin,z0,z1,ttau,tx)
      tauus2(nph)=tauus2(nph)+ttau
      xus2(nph)=xus2(nph)+tx
c     write(10,*)'is ks tauus2 xus2',is,ks,sngl(tauus2(nph)),
c    1 sngl(xus2(nph))
c
c   Take care of converted phases.
c
 41   iph=mod(nph,2)+1
      xus1(iph)=0d0
      xus2(iph)=0d0
      tauus1(iph)=0d0
      tauus2(iph)=0d0
      go to (59,61),nph
 61   if(umin.gt.pu(ku(1)+1,1)) go to 53
c
c   If we are doing an S-wave depth correction, we may need range and
c   tau for the P-wave which turns at the S-wave source slowness.  This
c   would bd needed for sPg and SPg when the source is in the deep mantle.
c
      do 44 j=1,nbrn
      if((phcd(j)(1:2).ne.'sP'.and.phcd(j)(1:2).ne.'SP').or.
     1 px(j,2).le.0d0) go to 44
c     write(10,*)'Depcor:  j phcd px umin =',j,' ',phcd(j),px(j,1),
c    1 px(j,2),umin
      if(umin.ge.px(j,1).and.umin.lt.px(j,2)) go to 45
 44   continue
      go to 53
c
c   If we are doing an P-wave depth correction, we may need range and
c   tau for the S-wave which turns at the P-wave source slowness.  This
c   would be needed for pS and PS.
c
 59   do 60 j=1,nbrn
      if((phcd(j)(1:2).ne.'pS'.and.phcd(j)(1:2).ne.'PS').or.
     1 px(j,2).le.0d0) go to 60
c     write(10,*)'Depcor:  j phcd px umin =',j,' ',phcd(j),px(j,1),
c    1 px(j,2),umin
      if(umin.ge.px(j,1).and.umin.lt.px(j,2)) go to 45
 60   continue
      go to 53
c
c   Do the integral.
 45   j=1
c     write(10,*)'Depcor:  do pS or sP integral - iph =',iph
      do 46 i=2,mt(iph)
      if(umin.ge.pm(i,iph)) go to 47
      call tauint(umin,pm(j,iph),pm(i,iph),zm(j,iph),zm(i,iph),ttau,tx)
      tauus1(iph)=tauus1(iph)+ttau
      xus1(iph)=xus1(iph)+tx
 46   j=i
 47   z1=zmod(umin,j,iph)
      if(dabs(zm(j,iph)-z1).le.dtol) go to 53
c   Unless the turning point is right on a sample slowness, one more
c   partial integral is needed.
      call tauint(umin,pm(j,iph),umin,zm(j,iph),z1,ttau,tx)
      tauus1(iph)=tauus1(iph)+ttau
      xus1(iph)=xus1(iph)+tx
c     write(10,*)'is ks tauusp xusp',j,ks,sngl(tauus1(iph)),
c    1 sngl(xus1(iph))
c
 53   ua(1,nph)=-1d0
c     if(odep.ge.deplim.or.odep.le..1) go to 43
      if(odep.ge.deplim) go to 43
      do 57 i=1,nseg
      if(.not.segmsk(i)) go to 57
      if(nafl(i,1).eq.nph.and.nafl(i,2).eq.0.and.iidx(i).le.0) go to 58
 57   continue
      go to 43
c
c   If the source is very shallow, we will need to insert some extra
c   ray parameter samples into the up-going branches.
c
 58   du=amin1(1e-5+(odep-.4)*2e-5,1e-5)
c     write(10,*)'Add:  nph is ka odep du us =',nph,is,ka,odep,
c    1 sngl(du),sngl(us(nph))
      lp=lpower
      k=0
      do 56 l=ka,1,-1
      k=k+1
      ua(k,nph)=us(nph)-(l**lp)*du
      lp=lp-1
      taua(k,nph)=0d0
      j=1
      if(is.lt.2) go to 54
      do 55 i=2,is
      call tauint(ua(k,nph),pm(j,nph),pm(i,nph),zm(j,nph),zm(i,nph),
     1 ttau,tx)
      taua(k,nph)=taua(k,nph)+ttau
 55   j=i
c     write(10,*)'l k ua taua',l,k,sngl(ua(k,nph)),sngl(taua(k,nph))
 54   if(dabs(zm(is,nph)-zs).le.dtol) go to 56
c   Unless the source is right on a sample slowness, one more partial
c   integral is needed.
      call tauint(ua(k,nph),pm(is,nph),us(nph),zm(is,nph),zs,ttau,tx)
      taua(k,nph)=taua(k,nph)+ttau
c     write(10,*)'l k ua taua',l,k,sngl(ua(k,nph)),sngl(taua(k,nph))
 56   continue
      go to 43
c
c   Construct tau for all branches.
c
 1    mu=mu+1
 43   j=1
c     write(10,*)'mu',mu
c     write(10,*)'killer loop:'
      do 20 i=1,nseg
      if(.not.segmsk(i)) go to 20
c     write(10,*)'i iidx nafl nph',i,iidx(i),nafl(i,1),nph
      if(iidx(i).gt.0.or.iabs(nafl(i,1)).ne.nph.or.(msrc(nph).le.0.and.
     1 nafl(i,1).gt.0)) go to 20
c
      iph=nafl(i,2)
      kph=nafl(i,3)
c   Handle up-going P and S.
      if(iph.le.0) iph=nph
      if(kph.le.0) kph=nph
      sgn=isign(1,nafl(i,1))
      i1=indx(i,1)
      i2=indx(i,2)
c     write(10,*)'i1 i2 sgn iph',i1,i2,sngl(sgn),iph
      m=1
      do 21 k=i1,i2
      if(pt(k).gt.umin) go to 22
 23   if(dabs(pt(k)-pu(m,nph)).le.dtol) go to 21
      m=m+1
      go to 23
 21   tau(1,k)=taut(k)+sgn*tauc(m)
      k=i2
c     write(10,*)'k m',k,m
      go to 24
c22   write(10,*)'k m',k,m
 22   if(dabs(pt(k-1)-umin).le.dtol) k=k-1
      ki=ki+1
      kk(ki)=k
      pk(ki)=pt(k)
      pt(k)=umin
      fac=fcs(i,1)
c     write(10,*)'ki fac',ki,sngl(fac)
      tau(1,k)=fac*(tauus1(iph)+tauus2(iph)+tauus1(kph)+tauus2(kph))+
     1 sgn*tauus1(nph)
c     write(10,*)'&&&&& nph iph kph tauus1 tauus2 tau =',
c    1 nph,iph,kph,sngl(tauus1(1)),sngl(tauus1(2)),sngl(tauus2(1)),
c    2 sngl(tauus2(2)),sngl(tau(1,k))
 24   m=1
 26   if(jndx(j,1).ge.indx(i,1)) go to 25
      j=j+1
      go to 26
 25   jndx(j,2)=min0(jidx(j),k)
      if(jndx(j,1).lt.jndx(j,2)) go to 37
      jndx(j,2)=-1
      go to 20
c37   write(10,*)'j jndx jidx',j,jndx(j,1),jndx(j,2),jidx(j),' ',
c    1 phcd(j)
 37   do 30 l=1,2
 28   if(dabs(pux(m,nph)-px(j,l)).le.dtol) go to 27
      if(m.ge.mu) go to 29
      m=m+1
      go to 28
 27   xbrn(j,l)=xt(j,l)+sgn*xc(m)
c     write(10,*)'x up:  j l m  ',j,l,m
      go to 30
 29   xbrn(j,l)=fac*(xus1(iph)+xus2(iph)+xus1(kph)+xus2(kph))+
     1 sgn*xus1(nph)
c     write(10,*)'x up:  j l end',j,l
c     write(10,*)'&&&&& nph iph kph xus1 xus2 xbrn =',
c    1 nph,iph,kph,sngl(xus1(1)),sngl(xus1(2)),sngl(xus2(1)),
c    2 sngl(xus2(2)),sngl(xbrn(j,l))
 30   continue
      if(j.ge.nbrn) go to 20
      j=j+1
      if(jndx(j,1).le.k) go to 25
 20   continue
      return
      end
      double precision function umod(zs,isrc,nph)
      save 
      include 'ttlim.inc'
      character*31 msg
      double precision pm,zm,us,pt,tau,xlim,xbrn,dbrn
      double precision zs,uend,dtol,zmod
      dimension isrc(2)
      common/umdc/pm(jsrc,2),zm(jsrc,2),ndex(jsrc,2),mt(2)
      common/tabc/us(2),pt(jout),tau(4,jout),xlim(2,jout),xbrn(jbrn,3),
     1 dbrn(jbrn,2),xn,pn,tn,dn,hn,jndx(jbrn,2),idel(jbrn,3),mbr1,mbr2
      data dtol/1d-6/
c
      m1=mt(nph)
      do 1 i=2,m1
      if(zm(i,nph).le.zs) go to 2
 1    continue
      dep=(1d0-dexp(zs))/xn
      write(msg,100)dep
      write(6,100)dep
 100  format('Source depth (',f6.1,') too deep.')
c     call abort(msg)
	write(*,*)msg
	call exit(0)
 2    if(dabs(zs-zm(i,nph)).le.dtol.and.dabs(zm(i,nph)-zm(i+1,nph)).le.
     1 dtol) go to 3
      j=i-1
      isrc(nph)=j
      umod=pm(j,nph)+(pm(i,nph)-pm(j,nph))*(dexp(zs-zm(j,nph))-1d0)/
     1 (dexp(zm(i,nph)-zm(j,nph))-1d0)
      return
 3    isrc(nph)=i
      umod=pm(i+1,nph)
      return
c
      entry zmod(uend,js,nph)
      i=js+1
      zmod=zm(js,nph)+dlog(dmax1((uend-pm(js,nph))*(dexp(zm(i,nph)-
     1 zm(js,nph))-1d0)/(pm(i,nph)-pm(js,nph))+1d0,1d-30))
      return
      end
      subroutine bkin(lu,nrec,len,buf)
c
c $$$$$ calls no other routines $$$$$
c
c   Bkin reads a block of len double precision words into array buf(len)
c   from record nrec of the direct access unformatted file connected to
c   logical unit lu.
c
      save
      double precision buf(len),tmp
c
      if(nrec.le.0) go to 1
      read(lu,rec=nrec)buf
      tmp=buf(1)
      return
c   If the record doesn't exist, zero fill the buffer.
 1    do 2 i=1,len
 2    buf(i)=0d0
      return
      end
      subroutine tauint(ptk,ptj,pti,zj,zi,tau,x)
      save
c
c $$$$$ calls warn $$$$$
c
c   Tauint evaluates the intercept (tau) and distance (x) integrals  for
c   the spherical earth assuming that slowness is linear between radii
c   for which the model is known.  The partial integrals are performed
c   for ray slowness ptk between model radii with slownesses ptj and pti
c   with equivalent flat earth depths zj and zi respectively.  The partial
c   integrals are returned in tau and x.  Note that ptk, ptj, pti, zj, zi,
c   tau, and x are all double precision.
c
      character*71 msg
      double precision ptk,ptj,pti,zj,zi,tau,x
      double precision xx,b,sqk,sqi,sqj,sqb
c
      if(dabs(zj-zi).le.1d-9) go to 13
      if(dabs(ptj-pti).gt.1d-9) go to 10
      if(dabs(ptk-pti).le.1d-9) go to 13
      b=dabs(zj-zi)
      sqj=dsqrt(dabs(ptj*ptj-ptk*ptk))
      tau=b*sqj
      x=b*ptk/sqj
      go to 4
 10   if(ptk.gt.1d-9.or.pti.gt.1d-9) go to 1
c   Handle the straight through ray.
      tau=ptj
      x=1.5707963267948966d0
      go to 4
 1    b=ptj-(pti-ptj)/(dexp(zi-zj)-1d0)
      if(ptk.gt.1d-9) go to 2
      tau=-(pti-ptj+b*dlog(pti/ptj)-b*dlog(dmax1((ptj-b)*pti/
     1 ((pti-b)*ptj),1d-30)))
      x=0d0
      go to 4
 2    if(ptk.eq.pti) go to 3
      if(ptk.eq.ptj) go to 11
      sqk=ptk*ptk
      sqi=dsqrt(dabs(pti*pti-sqk))
      sqj=dsqrt(dabs(ptj*ptj-sqk))
      sqb=dsqrt(dabs(b*b-sqk))
      if(sqb.gt.1d-30) go to 5
      xx=0d0
      x=ptk*(dsqrt(dabs((pti+b)/(pti-b)))-dsqrt(dabs((ptj+b)/
     1 (ptj-b))))/b
      go to 6
 5    if(b*b.lt.sqk) go to 7
      xx=dlog(dmax1((ptj-b)*(sqb*sqi+b*pti-sqk)/((pti-b)*
     1 (sqb*sqj+b*ptj-sqk)),1d-30))
      x=ptk*xx/sqb
      go to 6
 7    xx=dasin(dmax1(dmin1((b*pti-sqk)/(ptk*dabs(pti-b)),1d0),-1d0))-
     1 dasin(dmax1(dmin1((b*ptj-sqk)/(ptk*dabs(ptj-b)),1d0),-1d0))
      x=-ptk*xx/sqb
 6    tau=-(sqi-sqj+b*dlog((pti+sqi)/(ptj+sqj))-sqb*xx)
      go to 4
 3    sqk=pti*pti
      sqj=dsqrt(dabs(ptj*ptj-sqk))
      sqb=dsqrt(dabs(b*b-sqk))
      if(b*b.lt.sqk) go to 8
      xx=dlog(dmax1((ptj-b)*(b*pti-sqk)/((pti-b)*(sqb*sqj+b*ptj-sqk)),
     1 1d-30))
      x=pti*xx/sqb
      go to 9
 8    xx=dsign(1.5707963267948966d0,b-pti)-dasin(dmax1(dmin1((b*ptj-
     1 sqk)/(pti*dabs(ptj-b)),1d0),-1d0))
      x=-pti*xx/sqb
 9    tau=-(b*dlog(pti/(ptj+sqj))-sqj-sqb*xx)
      go to 4
 11   sqk=ptj*ptj
      sqi=dsqrt(dabs(pti*pti-sqk))
      sqb=dsqrt(dabs(b*b-sqk))
      if(b*b.lt.sqk) go to 12
      xx=dlog(dmax1((ptj-b)*(sqb*sqi+b*pti-sqk)/((pti-b)*(b*ptj-sqk)),
     1 1d-30))
      x=ptj*xx/sqb
      go to 14
 12   xx=dasin(dmax1(dmin1((b*pti-sqk)/(ptj*dabs(pti-b)),1d0),-1d0))-
     1 dsign(1.5707963267948966d0,b-ptj)
      x=-ptj*xx/sqb
 14   tau=-(b*dlog((pti+sqi)/ptj)+sqi-sqb*xx)
c
c   Handle various error conditions.
c
 4    if(x.ge.-1d-10) go to 15
      write(msg,100)ptk,ptj,pti,tau,x
 100  format('Bad range: ',1p5d12.4)
      call warn(msg)
 15   if(tau.ge.-1d-10) go to 16
      write(msg,101)ptk,ptj,pti,tau,x
 101  format('Bad tau: ',1p5d12.4)
      call warn(msg(1:69))
 16   return
c   Trap null integrals and handle them properly.
 13   tau=0d0
      x=0d0
      return
      end
      subroutine spfit(jb,int)
      save 
      include 'ttlim.inc'
      character*3 disc
      character*8 phcd
      logical newgrd,makgrd,segmsk,prnt
c     logical log
      double precision pm,zm,us,pt,tau,xlim,xbrn,dbrn,zs,pk,pu,pux,tauu,
     1 xu,px,xt,taut,coef,tauc,xc,tcoef,tp
      double precision pmn,dmn,dmx,hm,shm,thm,p0,p1,tau0,tau1,x0,x1,pe,
     1 pe0,spe0,scpe0,pe1,spe1,scpe1,dpe,dtau,dbrnch,cn,x180,x360,dtol,
     2 ptol,xmin,difpkp,xbot
      common/umdc/pm(jsrc,2),zm(jsrc,2),ndex(jsrc,2),mt(2)
      common/tabc/us(2),pt(jout),tau(4,jout),xlim(2,jout),xbrn(jbrn,3),
     1 dbrn(jbrn,2),xn,pn,tn,dn,hn,jndx(jbrn,2),idel(jbrn,3),mbr1,mbr2
      common/brkc/zs,pk(jseg),pu(jtsm0,2),pux(jxsm,2),tauu(jtsm,2),
     1 xu(jxsm,2),px(jbrn,2),xt(jbrn,2),taut(jout),coef(5,jout),
     2 tauc(jtsm),xc(jxsm),tcoef(5,jbrna,2),tp(jbrnu,2),odep,
     3 fcs(jseg,3),nin,nph0,int0(2),ki,msrc(2),isrc(2),nseg,nbrn,ku(2),
     4 km(2),nafl(jseg,3),indx(jseg,2),kndx(jseg,2),iidx(jseg),
     5 jidx(jbrn),kk(jseg)
      common/pcdc/phcd(jbrn)
      common/prtflc/segmsk(jseg),prnt(2)
      data dbrnch,cn,x180,x360,xmin,dtol,ptol/2.5307274d0,57.295779d0,
     1 3.1415927d0,6.2831853d0,3.92403d-3,1d-6,2d-6/
      data difpkp/3.1415926d0/
c
      if(prnt(1)) write(10,102)
      i1=jndx(jb,1)
      i2=jndx(jb,2)
c     write(10,*)'Spfit:  jb i1 i2 pt =',jb,i1,i2,sngl(pt(i1)),
c    1 sngl(pt(i2))
      if(i2-i1.gt.1.or.dabs(pt(i2)-pt(i1)).gt.ptol) go to 14
      jndx(jb,2)=-1
      return
 14   newgrd=.false.
      makgrd=.false.
      if(dabs(px(jb,2)-pt(i2)).gt.dtol) newgrd=.true.
c     write(10,*)'Spfit:  px newgrd =',sngl(px(jb,2)),newgrd
      if(.not.newgrd) go to 10
      k=mod(int-1,2)+1
      if(int.ne.int0(k)) makgrd=.true.
c     write(10,*)'Spfit:  int k int0 makgrd =',int,k,int0(k),makgrd
      if(int.gt.2) go to 12
c     call query('Enter xmin:',log)
c     read *,xmin
c     xmin=xmin*xn
      xmin=xn*amin1(amax1(2.*odep,2.),25.)
c     write(10,*)'Spfit:  xmin =',xmin,xmin/xn
      call pdecu(i1,i2,xbrn(jb,1),xbrn(jb,2),xmin,int,i2)
      jndx(jb,2)=i2
 12   nn=i2-i1+1
      if(makgrd) call tauspl(1,nn,pt(i1),tcoef(1,1,k))
c     write(10,301,iostat=ios)jb,k,nn,int,newgrd,makgrd,
c    1 xbrn(jb,1),xbrn(jb,2),(i,pt(i-1+i1),tau(1,i-1+i1),
c    2 (tcoef(j,i,k),j=1,5),i=1,nn)
c301  format(/1x,4i3,2l3,2f12.8/(1x,i5,0p2f12.8,1p5d10.2))
      call fitspl(1,nn,tau(1,i1),xbrn(jb,1),xbrn(jb,2),tcoef(1,1,k))
      int0(k)=int
      go to 11
 10   call fitspl(i1,i2,tau,xbrn(jb,1),xbrn(jb,2),coef)
 11   pmn=pt(i1)
      dmn=xbrn(jb,1)
      dmx=dmn
      mxcnt=0
      mncnt=0
c     call appx(i1,i2,xbrn(jb,1),xbrn(jb,2))
c     write(10,300)(i,pt(i),(tau(j,i),j=1,3),i=i1,i2)
c300  format(/(1x,i5,4f12.6))
      pe=pt(i2)
      p1=pt(i1)
      tau1=tau(1,i1)
      xbot=tau(2,i1)
      x1=xbot
      pe1=pe-p1
      spe1=dsqrt(dabs(pe1))
      scpe1=pe1*spe1
      j=i1
      is=i1+1
      do 2 i=is,i2
      p0=p1
      p1=pt(i)
      tau0=tau1
      tau1=tau(1,i)
      x0=x1
      x1=tau(2,i)
      dpe=p0-p1
      dtau=tau1-tau0
      pe0=pe1
      pe1=pe-p1
      spe0=spe1
      spe1=dsqrt(dabs(pe1))
      scpe0=scpe1
      scpe1=pe1*spe1
      tau(4,j)=(2d0*dtau-dpe*(x1+x0))/(.5d0*(scpe1-scpe0)-1.5d0*spe1*
     1 spe0*(spe1-spe0))
      tau(3,j)=(dtau-dpe*x0-(scpe1+.5d0*scpe0-1.5d0*pe1*spe0)*tau(4,j))/
     1 (dpe*dpe)
      tau(2,j)=(dtau-(pe1*pe1-pe0*pe0)*tau(3,j)-(scpe1-scpe0)*tau(4,j))/
     1 dpe
      tau(1,j)=tau0-scpe0*tau(4,j)-pe0*(pe0*tau(3,j)+tau(2,j))
      xlim(1,j)=dmin1(x0,x1)
      xlim(2,j)=dmax1(x0,x1)
      if(xlim(1,j).ge.dmn) go to 5
      dmn=xlim(1,j)
      pmn=pt(j)
      if(x1.lt.x0) pmn=pt(i)
 5    disc=' '
      if(dabs(tau(3,j)).le.1d-30) go to 4
      shm=-.375d0*tau(4,j)/tau(3,j)
      hm=shm*shm
      if(shm.le.0d0.or.(hm.le.pe1.or.hm.ge.pe0)) go to 4
 7    thm=tau(2,j)+shm*(2d0*shm*tau(3,j)+1.5d0*tau(4,j))
      xlim(1,j)=dmin1(xlim(1,j),thm)
      xlim(2,j)=dmax1(xlim(2,j),thm)
      if(thm.ge.dmn) go to 6
      dmn=thm
      pmn=pe-hm
 6    disc='max'
      if(tau(4,j).lt.0d0) disc='min'
      if(disc.eq.'max') mxcnt=mxcnt+1
      if(disc.eq.'min') mncnt=mncnt+1
 4    if(prnt(1)) write(10,100,iostat=ios)disc,j,pt(j),
     1 (tau(k,j),k=1,4),(cn*xlim(k,j),k=1,2)
 100  format(1x,a,i5,f10.6,1p4e10.2,0p2f7.2)
      dmx=dmax1(dmx,xlim(2,j))
 2    j=i
c     if(prnt(1)) write(10,100,iostat=ios)'   ',j,pt(j)
      xbrn(jb,1)=dmn
      xbrn(jb,2)=dmx
      xbrn(jb,3)=pmn
      idel(jb,1)=1
      idel(jb,2)=1
      if(xbrn(jb,1).gt.x180) idel(jb,1)=2
      if(xbrn(jb,2).gt.x180) idel(jb,2)=2
      if(xbrn(jb,1).gt.x360) idel(jb,1)=3
      if(xbrn(jb,2).gt.x360) idel(jb,2)=3
      if(int.gt.2) go to 1
      phcd(jb)=phcd(jb)(1:1)
      i=jb
      do 8 j=1,nbrn
      i=mod(i,nbrn)+1
      if(phcd(i)(1:1).eq.phcd(jb).and.phcd(i)(2:2).ne.'P'.and.
     1 (pe.ge.px(i,1).and.pe.le.px(i,2))) go to 9
 8    continue
      go to 1
 9    phcd(jb)=phcd(i)
      if(dabs(pt(i2)-pt(jndx(i,1))).le.dtol) phcd(jb)=phcd(i-1)
 1    if(prnt(1).and.prnt(2)) write(10,102)
 102  format()
      if(dbrn(jb,1).le.0d0) go to 3
      dbrn(jb,1)=xbot
      dbrn(jb,2)=dbrnch
      if(index(phcd(jb),'ab').gt.0) dbrn(jb,2)=difpkp
      if(prnt(2)) write(10,101,iostat=ios)phcd(jb),
     1 (jndx(jb,k),k=1,2),(cn*xbrn(jb,k),k=1,2),xbrn(jb,3),
     2 (cn*dbrn(jb,k),k=1,2),(idel(jb,k),k=1,3),int,newgrd,makgrd
 101  format(1x,a,2i5,2f8.2,f8.4,2f8.2,4i3,2l2)
      go to 15
 3    if(prnt(2)) write(10,103,iostat=ios)phcd(jb),
     1 (jndx(jb,k),k=1,2),(cn*xbrn(jb,k),k=1,2),xbrn(jb,3),
     2 (idel(jb,k),k=1,3),int,newgrd,makgrd
 103  format(1x,a,2i5,2f8.2,f8.4,16x,4i3,2l2)
 15   if(mxcnt.gt.mncnt.or.mncnt.gt.mxcnt+1)
     1 call warn('Bad interpolation on '//phcd(jb))
      return
      end
      subroutine pdecu(i1,i2,x0,x1,xmin,int,len)
      save 
      include 'ttlim.inc'
      double precision us,pt,tau,xlim,xbrn,dbrn,ua,taua
      double precision x0,x1,xmin,dx,dx2,sgn,rnd,xm,axm,x,h1,h2,hh,xs
      common/tabc/us(2),pt(jout),tau(4,jout),xlim(2,jout),xbrn(jbrn,3),
     1 dbrn(jbrn,2),xn,pn,tn,dn,hn,jndx(jbrn,2),idel(jbrn,3),mbr1,mbr2
      common/pdec/ua(5,2),taua(5,2),deplim,ka
c
c     write(10,*)'Pdecu:  ua =',sngl(ua(1,int))
      if(ua(1,int).le.0d0) go to 17
c     write(10,*)'Pdecu:  fill in new grid'
      k=i1+1
      do 18 i=1,ka
      pt(k)=ua(i,int)
      tau(1,k)=taua(i,int)
 18   k=k+1
      pt(k)=pt(i2)
      tau(1,k)=tau(1,i2)
      go to 19
c
 17   is=i1+1
      ie=i2-1
      xs=x1
      do 11 i=ie,i1,-1
      x=xs
      if(i.ne.i1) go to 12
      xs=x0
      go to 14
 12   h1=pt(i-1)-pt(i)
      h2=pt(i+1)-pt(i)
      hh=h1*h2*(h1-h2)
      h1=h1*h1
      h2=-h2*h2
      xs=-(h2*tau(1,i-1)-(h2+h1)*tau(1,i)+h1*tau(1,i+1))/hh
 14   if(dabs(x-xs).le.xmin) go to 15
 11   continue
      len=i2
      return
 15   ie=i
      if(dabs(x-xs).gt..75d0*xmin.or.ie.eq.i2) go to 16
      xs=x
      ie=ie+1
 16   n=max0(idint(dabs(xs-x0)/xmin+.8d0),1)
      dx=(xs-x0)/n
      dx2=dabs(.5d0*dx)
      sgn=dsign(1d0,dx)
      rnd=0d0
      if(sgn.gt.0d0) rnd=1d0
      xm=x0+dx
      k=i1
      m=is
      axm=1d10
      do 1 i=is,ie
      if(i.lt.ie) go to 8
      x=xs
      go to 5
 8    h1=pt(i-1)-pt(i)
      h2=pt(i+1)-pt(i)
      hh=h1*h2*(h1-h2)
      h1=h1*h1
      h2=-h2*h2
      x=-(h2*tau(1,i-1)-(h2+h1)*tau(1,i)+h1*tau(1,i+1))/hh
 5    if(sgn*(x-xm).le.dx2) go to 2
      if(k.lt.m) go to 3
      do 4 j=m,k
 4    pt(j)=-1d0
 3    m=k+2
      k=i-1
      axm=1d10
 7    xm=xm+dx*idint((x-xm-dx2)/dx+rnd)
 2    if(dabs(x-xm).ge.axm) go to 1
      axm=dabs(x-xm)
      k=i-1
 1    continue
      if(k.lt.m) go to 9
      do 6 j=m,k
 6    pt(j)=-1d0
 9    k=i1
      do 10 i=is,i2
      if(pt(i).lt.0d0) go to 10
      k=k+1
      pt(k)=pt(i)
      tau(1,k)=tau(1,i)
 10   continue
 19   len=k
c     write(10,300)(i,pt(i),tau(1,i),i=i1,len)
c300  format(/(1x,i5,0pf12.6,1pd15.4))
      return
      end
      subroutine tauspl(i1,i2,pt,coef)
c
c $$$$$ calls only library routines $$$$$
c
c   Given ray parameter grid pt;i (pt sub i), i=i1,i1+1,...,i2, tauspl
c   determines the i2-i1+3 basis functions for interpolation I such
c   that:
c
c      tau(p) = a;1,i + Dp * a;2,i + Dp**2 * a;3,i + Dp**(3/2) * a;4,i
c
c   where Dp = pt;n - p, pt;i <= p < pt;i+1, and the a;j,i's are
c   interpolation coefficients.  Rather than returning the coefficients,
c   a;j,i, which necessarily depend on tau(pt;i), i=i1,i1+1,...,i2 and
c   x(pt;i) (= -d tau(p)/d p | pt;i), i=i1,i2, tauspl returns the
c   contribution of each basis function and its derivitive at each
c   sample.  Each basis function is non-zero at three grid points,
c   therefore, each grid point will have contributions (function values
c   and derivitives) from three basis functions.  Due to the basis
c   function normalization, one of the function values will always be
c   one and is not returned in array coef with the other values.
c   Rewritten on 23 December 1983 by R. Buland.
c
      save
      double precision pt(i2),coef(5,i2)
      double precision del(5),sdel(5),deli(5),d3h(4),d1h(4),dih(4),
     1 d(4),ali,alr,b3h,b1h,bih,th0p,th2p,th3p,th2m
c
      n2=i2-i1-1
      if(n2.le.-1) return
      is=i1+1
c
c   To achieve the requisite stability, proceed by constructing basis
c   functions G;i, i=0,1,...,n+1.  G;i will be non-zero only on the
c   interval [p;i-2,p;i+2] and will be continuous with continuous first
c   and second derivitives.  G;i(p;i-2) and G;i(p;i+2) are constrained
c   to be zero with zero first and second derivitives.  G;i(p;i) is
c   normalized to unity.
c
c   Set up temporary variables appropriate for G;-1.  Note that to get
c   started, the ray parameter grid is extrapolated to yeild p;i, i=-2,
c   -1,0,1,...,n.
      del(2)=pt(i2)-pt(i1)+3d0*(pt(is)-pt(i1))
      sdel(2)=dsqrt(dabs(del(2)))
      deli(2)=1d0/sdel(2)
      m=2
      do 1 k=3,5
      del(k)=pt(i2)-pt(i1)+(5-k)*(pt(is)-pt(i1))
      sdel(k)=dsqrt(dabs(del(k)))
      deli(k)=1d0/sdel(k)
      d3h(m)=del(k)*sdel(k)-del(m)*sdel(m)
      d1h(m)=sdel(k)-sdel(m)
      dih(m)=deli(k)-deli(m)
 1    m=k
      l=i1-1
      if(n2.le.0) go to 10
c   Loop over G;i, i=0,1,...,n-3.
      do 2 i=1,n2
      m=1
c   Update temporary variables for G;i-1.
      do 3 k=2,5
      del(m)=del(k)
      sdel(m)=sdel(k)
      deli(m)=deli(k)
      if(k.ge.5) go to 3
      d3h(m)=d3h(k)
      d1h(m)=d1h(k)
      dih(m)=dih(k)
 3    m=k
      l=l+1
      del(5)=pt(i2)-pt(l+1)
      sdel(5)=dsqrt(dabs(del(5)))
      deli(5)=1d0/sdel(5)
      d3h(4)=del(5)*sdel(5)-del(4)*sdel(4)
      d1h(4)=sdel(5)-sdel(4)
      dih(4)=deli(5)-deli(4)
c   Construct G;i-1.
      ali=1d0/(.125d0*d3h(1)-(.75d0*d1h(1)+.375d0*dih(1)*del(3))*
     1 del(3))
      alr=ali*(.125d0*del(2)*sdel(2)-(.75d0*sdel(2)+.375d0*del(3)*
     1 deli(2)-sdel(3))*del(3))
      b3h=d3h(2)+alr*d3h(1)
      b1h=d1h(2)+alr*d1h(1)
      bih=dih(2)+alr*dih(1)
      th0p=d1h(1)*b3h-d3h(1)*b1h
      th2p=d1h(3)*b3h-d3h(3)*b1h
      th3p=d1h(4)*b3h-d3h(4)*b1h
      th2m=dih(3)*b3h-d3h(3)*bih
c   The d;i's completely define G;i-1.
      d(4)=ali*((dih(1)*b3h-d3h(1)*bih)*th2p-th2m*th0p)/((dih(4)*b3h-
     1 d3h(4)*bih)*th2p-th2m*th3p)
      d(3)=(th0p*ali-th3p*d(4))/th2p
      d(2)=(d3h(1)*ali-d3h(3)*d(3)-d3h(4)*d(4))/b3h
      d(1)=alr*d(2)-ali
c   Construct the contributions G;i-1(p;i-2) and G;i-1(p;i).
c   G;i-1(p;i-1) need not be constructed as it is normalized to unity.
      coef(1,l)=(.125d0*del(5)*sdel(5)-(.75d0*sdel(5)+.375d0*deli(5)*
     1 del(4)-sdel(4))*del(4))*d(4)
      if(i.ge.3) coef(2,l-2)=(.125d0*del(1)*sdel(1)-(.75d0*sdel(1)+
     1 .375d0*deli(1)*del(2)-sdel(2))*del(2))*d(1)
c   Construct the contributions -dG;i-1(p)/dp | p;i-2, p;i-1, and p;i.
      coef(3,l)=-.75d0*(sdel(5)+deli(5)*del(4)-2d0*sdel(4))*d(4)
      if(i.ge.2) coef(4,l-1)=-.75d0*((sdel(2)+deli(2)*del(3)-
     1 2d0*sdel(3))*d(2)-(d1h(1)+dih(1)*del(3))*d(1))
      if(i.ge.3) coef(5,l-2)=-.75d0*(sdel(1)+deli(1)*del(2)-
     1 2d0*sdel(2))*d(1)
 2    continue
c   Loop over G;i, i=n-2,n-1,n,n+1.  These cases must be handled
c   seperately because of the singularities in the second derivitive
c   at p;n.
 10   do 4 j=1,4
      m=1
c   Update temporary variables for G;i-1.
      do 5 k=2,5
      del(m)=del(k)
      sdel(m)=sdel(k)
      deli(m)=deli(k)
      if(k.ge.5) go to 5
      d3h(m)=d3h(k)
      d1h(m)=d1h(k)
      dih(m)=dih(k)
 5    m=k
      l=l+1
      del(5)=0d0
      sdel(5)=0d0
      deli(5)=0d0
c   Construction of the d;i's is different for each case.  In cases
c   G;i, i=n-1,n,n+1, G;i is truncated at p;n to avoid patching across
c   the singularity in the second derivitive.
      if(j.lt.4) go to 6
c   For G;n+1 constrain G;n+1(p;n) to be .25.
      d(1)=2d0/(del(1)*sdel(1))
      go to 9
c   For G;i, i=n-2,n-1,n, the condition dG;i(p)/dp|p;i = 0 has been
c   substituted for the second derivitive continuity condition that
c   can no longer be satisfied.
 6    alr=(sdel(2)+deli(2)*del(3)-2d0*sdel(3))/(d1h(1)+dih(1)*del(3))
      d(2)=1d0/(.125d0*del(2)*sdel(2)-(.75d0*sdel(2)+.375d0*deli(2)*
     1 del(3)-sdel(3))*del(3)-(.125d0*d3h(1)-(.75d0*d1h(1)+.375d0*
     2 dih(1)*del(3))*del(3))*alr)
      d(1)=alr*d(2)
      if(j-2)8,7,9
c   For G;n-1 constrain G;n-1(p;n) to be .25.
 7    d(3)=(2d0+d3h(2)*d(2)+d3h(1)*d(1))/(del(3)*sdel(3))
      go to 9
c   No additional constraints are required for G;n-2.
 8    d(3)=-((d3h(2)-d1h(2)*del(4))*d(2)+(d3h(1)-d1h(1)*del(4))*
     1 d(1))/(d3h(3)-d1h(3)*del(4))
      d(4)=(d3h(3)*d(3)+d3h(2)*d(2)+d3h(1)*d(1))/(del(4)*sdel(4))
c   Construct the contributions G;i-1(p;i-2) and G;i-1(p;i).
 9    if(j.le.2) coef(1,l)=(.125d0*del(3)*sdel(3)-(.75d0*sdel(3)+.375d0*
     1 deli(3)*del(4)-sdel(4))*del(4))*d(3)-(.125d0*d3h(2)-(.75d0*
     2 d1h(2)+.375d0*dih(2)*del(4))*del(4))*d(2)-(.125d0*d3h(1)-(.75d0*
     3 d1h(1)+.375d0*dih(1)*del(4))*del(4))*d(1)
      if(l-i1.gt.1) coef(2,l-2)=(.125d0*del(1)*sdel(1)-(.75d0*sdel(1)+
     1 .375d0*deli(1)*del(2)-sdel(2))*del(2))*d(1)
c   Construct the contributions -dG;i-1(p)/dp | p;i-2, p;i-1, and p;i.
      if(j.le.2) coef(3,l)=-.75d0*((sdel(3)+deli(3)*del(4)-
     1 2d0*sdel(4))*d(3)-(d1h(2)+dih(2)*del(4))*d(2)-(d1h(1)+
     2 dih(1)*del(4))*d(1))
      if(j.le.3.and.l-i1.gt.0) coef(4,l-1)=0d0
      if(l-i1.gt.1) coef(5,l-2)=-.75d0*(sdel(1)+deli(1)*del(2)-
     1 2d0*sdel(2))*d(1)
 4    continue
      return
      end
      subroutine fitspl(i1,i2,tau,x1,xn,coef)
c
c $$$$$ calls only library routines $$$$$
c
c   Given ray parameter grid p;i (p sub i), i=1,2,...,n, corresponding
c   tau;i values, and x;1 and x;n (x;i = -dtau/dp|p;i); tauspl finds
c   interpolation I such that:  tau(p) = a;1,i + Dp * a;2,i + Dp**2 *
c   a;3,i + Dp**(3/2) * a;4,i where Dp = p;n - p and p;i <= p < p;i+1.
c   Interpolation I has the following properties:  1) x;1, x;n, and
c   tau;i, i=1,2,...,n are fit exactly, 2) the first and second
c   derivitives with respect to p are continuous everywhere, and
c   3) because of the paramaterization d**2 tau/dp**2|p;n is infinite.
c   Thus, interpolation I models the asymptotic behavior of tau(p)
c   when tau(p;n) is a branch end due to a discontinuity in the
c   velocity model.  Note that array a must be dimensioned at least
c   a(4,n) though the interpolation coefficients will be returned in
c   the first n-1 columns.  The remaining column is used as scratch
c   space and returned as all zeros.  Programmed on 16 August 1982 by
c   R. Buland.
c
      save 
      double precision tau(4,i2),x1,xn,coef(5,i2),a(2,100),ap(3),
     1 b(100),alr,g1,gn
c
      if(i2-i1)13,1,2
 1    tau(2,i1)=x1
 13   return
 2    n=0
      do 3 i=i1,i2
      n=n+1
      b(n)=tau(1,i)
      do 3 j=1,2
 3    a(j,n)=coef(j,i)
      do 4 j=1,3
 4    ap(j)=coef(j+2,i2)
      n1=n-1
c
c   Arrays ap(*,1), a, and ap(*,2) comprise n+2 x n+2 penta-diagonal
c   matrix A.  Let x1, tau, and xn comprise corresponding n+2 vector b.
c   Then, A * g = b, may be solved for n+2 vector g such that
c   interpolation I is given by I(p) = sum(i=0,n+1) g;i * G;i(p).
c
c   Eliminate the lower triangular portion of A to form A'.  A
c   corresponding transformation applied to vector b is stored in
c   a(4,*).
      alr=a(1,1)/coef(3,i1)
      a(1,1)=1d0-coef(4,i1)*alr
      a(2,1)=a(2,1)-coef(5,i1)*alr
      b(1)=b(1)-x1*alr
      j=1
      do 5 i=2,n
      alr=a(1,i)/a(1,j)
      a(1,i)=1d0-a(2,j)*alr
      b(i)=b(i)-b(j)*alr
 5    j=i
      alr=ap(1)/a(1,n1)
      ap(2)=ap(2)-a(2,n1)*alr
      gn=xn-b(n1)*alr
      alr=ap(2)/a(1,n)
c   Back solve the upper triangular portion of A' for coefficients g;i.
c   When finished, storage g(2), a(4,*), g(5) will comprise vector g.
      gn=(gn-b(n)*alr)/(ap(3)-a(2,n)*alr)
      b(n)=(b(n)-gn*a(2,n))/a(1,n)
      j=n
      do 6 i=n1,1,-1
      b(i)=(b(i)-b(j)*a(2,i))/a(1,i)
 6    j=i
      g1=(x1-coef(4,i1)*b(1)-coef(5,i1)*b(2))/coef(3,i1)
c
      tau(2,i1)=x1
      is=i1+1
      ie=i2-1
      j=1
      do 7 i=is,ie
      j=j+1
 7    tau(2,i)=coef(3,i)*b(j-1)+coef(4,i)*b(j)+coef(5,i)*b(j+1)
      tau(2,i2)=xn
      return
      end
      subroutine trtm(delta,max,n,tt,dtdd,dtdh,dddp,phnm)
      save 
      include 'ttlim.inc'
      character*(*) phnm(max)
      character*8 ctmp(60)
      dimension tt(max),dtdd(max),dtdh(max),dddp(max),tmp(60,4),
     1 iptr(60)
      double precision us,pt,tau,xlim,xbrn,dbrn
      double precision x(3),cn,dtol,pi,pi2
      common/tabc/us(2),pt(jout),tau(4,jout),xlim(2,jout),xbrn(jbrn,3),
     1 dbrn(jbrn,2),xn,pn,tn,dn,hn,jndx(jbrn,2),idel(jbrn,3),mbr1,mbr2
      data cn,dtol,atol,pi,pi2/.017453292519943296d0,1d-6,.005,
     1 3.1415926535897932d0,6.2831853071795865d0/
c
      n=0
      if(mbr2.le.0) return
      x(1)=dmod(dabs(cn*delta),pi2)
      if(x(1).gt.pi) x(1)=pi2-x(1)
      x(2)=pi2-x(1)
      x(3)=x(1)+pi2
      if(dabs(x(1)).gt.dtol) go to 9
      x(1)=dtol
      x(3)=-10d0
 9    if(dabs(x(1)-pi).gt.dtol) go to 7
      x(1)=pi-dtol
      x(2)=-10d0
 7    do 1 j=mbr1,mbr2
 1    if(jndx(j,2).gt.0) call findtt(j,x,max,n,tmp,tmp(1,2),tmp(1,3),
     1 tmp(1,4),ctmp)
      if(n-1)3,4,5
 4    iptr(1)=1
      go to 6
 5    call r4sort(n,tmp,iptr)
 6    k=0
      do 2 i=1,n
      j=iptr(i)
      if(k.le.0) go to 8
      if(phnm(k).eq.ctmp(j).and.abs(tt(k)-tmp(j,1)).le.atol) go to 2
 8    k=k+1
      tt(k)=tmp(j,1)
      dtdd(k)=tmp(j,2)
      dtdh(k)=tmp(j,3)
      dddp(k)=tmp(j,4)
      phnm(k)=ctmp(j)
 2    continue
      n=k
 3    return
      end
      subroutine findtt(jb,x0,max,n,tt,dtdd,dtdh,dddp,phnm)
      save 
      include 'ttlim.inc'
      character*(*) phnm(max)
      character*8 phcd
      character*67 msg
      dimension tt(max),dtdd(max),dtdh(max),dddp(max)
      double precision us,pt,tau,xlim,xbrn,dbrn
      double precision x,x0(3),p0,p1,arg,dp,dps,delp,tol,ps,deps
      common/tabc/us(2),pt(jout),tau(4,jout),xlim(2,jout),xbrn(jbrn,3),
     1 dbrn(jbrn,2),xn,pn,tn,dn,hn,jndx(jbrn,2),idel(jbrn,3),mbr1,mbr2
      common/pcdc/phcd(jbrn)
      data tol/3d-6/,deps/1d-10/
c
      nph=iabs(idel(jb,3))
      hsgn=isign(1,idel(jb,3))*hn
      dsgn=(-1.)**idel(jb,1)*dn
      dpn=-1./tn
      do 10 ij=idel(jb,1),idel(jb,2)
      x=x0(ij)
      dsgn=-dsgn
      if(x.lt.xbrn(jb,1).or.x.gt.xbrn(jb,2)) go to 12
      j=jndx(jb,1)
      is=j+1
      ie=jndx(jb,2)
      do 1 i=is,ie
      if(x.le.xlim(1,j).or.x.gt.xlim(2,j)) go to 8
      le=n
      p0=pt(ie)-pt(j)
      p1=pt(ie)-pt(i)
      delp=dmax1(tol*(pt(i)-pt(j)),1d-3)
      if(dabs(tau(3,j)).gt.1d-30) go to 2
      dps=(x-tau(2,j))/(1.5d0*tau(4,j))
      dp=dsign(dps*dps,dps)
      dp0=dp
      if(dp.lt.p1-delp.or.dp.gt.p0+delp) go to 9
      if(n.ge.max) go to 13
      n=n+1
      ps=pt(ie)-dp
      tt(n)=tn*(tau(1,j)+dp*(tau(2,j)+dps*tau(4,j))+ps*x)
      dtdd(n)=dsgn*ps
      dtdh(n)=hsgn*sqrt(abs(sngl(us(nph)*us(nph)-ps*ps)))
      dddp(n)=dpn*.75d0*tau(4,j)/dmax1(dabs(dps),deps)
      phnm(n)=phcd(jb)
      in=index(phnm(n),'ab')
      if(in.le.0) go to 8
      if(ps.le.xbrn(jb,3)) phnm(n)(in:)='bc'
      go to 8
 2    do 4 jj=1,2
      go to (5,6),jj
 5    arg=9d0*tau(4,j)*tau(4,j)+32d0*tau(3,j)*(x-tau(2,j))
      if(arg.ge.0d0) go to 3
      write(msg,100)arg
 100  format('Bad sqrt argument:',1pd11.2,'.')
      call warn(msg(1:30))
 3    dps=-(3d0*tau(4,j)+dsign(dsqrt(dabs(arg)),tau(4,j)))/(8d0*
     1 tau(3,j))
      dp=dsign(dps*dps,dps)
      dp0=dp
      go to 7
 6    dps=(tau(2,j)-x)/(2d0*tau(3,j)*dps)
      dp=dsign(dps*dps,dps)
 7    if(dp.lt.p1-delp.or.dp.gt.p0+delp) go to 4
      if(n.ge.max) go to 13
      n=n+1
      ps=pt(ie)-dp
      tt(n)=tn*(tau(1,j)+dp*(tau(2,j)+dp*tau(3,j)+dps*tau(4,j))+ps*x)
      dtdd(n)=dsgn*ps
      dtdh(n)=hsgn*sqrt(abs(sngl(us(nph)*us(nph)-ps*ps)))
      dddp(n)=dpn*(2d0*tau(3,j)+.75d0*tau(4,j)/dmax1(dabs(dps),deps))
      phnm(n)=phcd(jb)
      in=index(phnm(n),'ab')
      if(in.le.0) go to 4
      if(ps.le.xbrn(jb,3)) phnm(n)(in:)='bc'
 4    continue
 9    if(n.gt.le) go to 8
      write(msg,101)phcd(jb),x,dp0,dp,p1,p0
 101  format('Failed to find phase:  ',a,f8.1,4f7.4)
      call warn(msg)
 8    j=i
 1    continue
c
 12   if(x.lt.dbrn(jb,1).or.x.gt.dbrn(jb,2)) go to 10
      if(n.ge.max) go to 13
      j=jndx(jb,1)
      i=jndx(jb,2)
      dp=pt(i)-pt(j)
      dps=dsqrt(dabs(dp))
      n=n+1
      tt(n)=tn*(tau(1,j)+dp*(tau(2,j)+dp*tau(3,j)+dps*tau(4,j))+
     1 pt(j)*x)
      dtdd(n)=dsgn*sngl(pt(j))
      dtdh(n)=hsgn*sqrt(abs(sngl(us(nph)*us(nph)-pt(j)*pt(j))))
      dddp(n)=dpn*(2d0*tau(3,j)+.75d0*tau(4,j)/dmax1(dps,deps))
      ln=index(phcd(jb),'ab')-1
      if(ln.le.0) ln=index(phcd(jb),' ')-1
      if(ln.le.0) ln=len(phcd(jb))
      phnm(n)=phcd(jb)(1:ln)//'diff'
 10   continue
      return
 13   write(msg,102)max
 102  format('More than',i3,' arrivals found.')
      call warn(msg(1:28))
      return
      end
      function iargcX(i)
c
c $$$$$ calls only MSDOS library routines $$$$$
c
c   IargcX emulates the UNIX command iargc using the MSDOS
c   nargs equivalent.
c
c     i=nargs()-1
	i=iargc()-1
      iargcX=i
      return
      end
      subroutine asnag1(lu,mode,n,ia,ib)
c
c $$$$$ calls assign, iargcX, and getarg $$$$$
c
c   Asnag1 assigns logical unit lu to a direct access disk file
c   with mode "mode" and record length "len".  See dasign for 
c   details.  The n th argument is used as the model name.  If there 
c   is no n th argument and ib is non-blank, it is taken to be the 
c   model name.  If ib is blank, the user is prompted for the
c   model name using the character string in variable ia as the
c   prompt.  Programmed on 8 October 1980 by R. Buland.
c
      save
      logical log
      character*(*) ia,ib
      character*500 filename
c
      if(iargcX(i).lt.n) go to 1
      call getarg(n,ib)
      go to 2
c
 1    if(ib.ne.' ') go to 2
      call query(ia,log)
      read(*,100)ib
 100  format(a)
c
 2    nb=lenc(ib)
 	filename = ib(1:nb)//'.hed'
      call assign(lu,mode,filename)
      return
      end
      subroutine assign(lu,mode,ia)
c
c $$$$$ calls no other routine $$$$$
c
c   Subroutine assign opens (connects) logical unit lu to the disk file
c   named by the character string ia with mode mode.  If iabs(mode) = 1,
c   then open the file for reading.  If iabs(mode) = 2, then open the
c   file for writing.  If iabs(mode) = 3, then open a scratch file for
c   writing.  If mode > 0, then the file is formatted.  If mode < 0,
c   then the file is unformatted.  All files opened by assign are
c   assumed to be sequential.  Programmed on 3 December 1979 by
c   R. Buland.
c
      save
      character*(*) ia
      logical exst
c
      if(mode.ge.0) nf=1
      if(mode.lt.0) nf=2
      ns=iabs(mode)
      if(ns.le.0.or.ns.gt.3) ns=3
      go to (1,2),nf
 1    go to (11,12,13),ns
 11   open(lu,file=ia,status='old',form='formatted')
      rewind lu
      return
 12   inquire(file=ia,exist=exst)
      if(exst) go to 11
 13   open(lu,file=ia,status='new',form='formatted')
      return
 2    go to (21,22,23),ns
 21   open(lu,file=ia,status='old',form='unformatted')
      rewind lu
      return
 22   inquire(file=ia,exist=exst)
      if(exst) go to 21
 23   open(lu,file=ia,status='new',form='unformatted')
      return
      end
      subroutine retrns(lu)
c
c $$$$$ calls no other routine $$$$$
c
c   Subroutine retrns closes (disconnects) logical unit lu from the
c   calling program.  Programmed on 3 December 1979 by R. Buland.
c
      save
      close(unit=lu)
      return
      end
      subroutine query(ia,log)
c
c $$$$$ calls tnoua $$$$$
c
c   Subroutine query scans character string ia (up to 78 characters) for
c   a question mark or a colon.  It prints the string up to and
c   including the flag character plus two blanks with no newline on the
c   standard output.  If the flag was a question mark, query reads the
c   users response.  If the response is 'y' or 'yes', log is set to
c   true.  If the response is 'n' or 'no', log is set to false.  Any
c   other response causes the question to be repeated.  If the flag was
c   a colon, query simply returns allowing user input on the same line.
c   If there is no question mark or colon, the last non-blank character
c   is treated as if it were a colon.  If the string is null or all
c   blank, query prints an astrisk and returns.  Programmed on 3
c   December 1979 by R. Buland.
c
      save
      logical log
      character*(*) ia
      character*81 ib
      character*4 ans
      nn=len(ia)
      log=.true.
      ifl=1
      k=0
c   Scan ia for flag characters or end-of-string.
      do 1 i=1,nn
      ib(i:i)=ia(i:i)
      if(ib(i:i).eq.':') go to 7
      if(ib(i:i).eq.'?') go to 3
      if(ib(i:i).eq.'\0') go to 5
      if(ib(i:i).ne.' ') k=i
 1    continue
c   If we fell off the end of the string, branch if there were any non-
c   blank characters.
 5    if(k.gt.0) go to 6
c   Handle a null or all blank string.
      i=1
      ib(i:i)='*'
      go to 4
c   Handle a string with no question mark or colon but at least one
c   non-blank character.
 6    i=k
c   Append two blanks and print the string.
 7    i=i+2
      ib(i-1:i-1)=' '
      ib(i:i)=' '
c   Tnoua prints the first i characters of ib without a newline.
 4    call tnoua(ib,i)
      if(ifl.gt.0) return
c   If the string was a yes-no question read the response.
      read 102,ans
 102  format(a4)
      call uctolc(ans,-1)
c   If the response is yes log is already set properly.
      if(ans.eq.'y   '.or.ans.eq.'yes ') return
c   If the response is no set log to false.  Otherwise repeat the
c   question.
      if(ans.ne.'n   '.and.ans.ne.'no  ') go to 4
      log=.false.
      return
 3    ifl=-ifl
      go to 7
      end
      subroutine uctolc(ia,ifl)
c
c $$$$$ calls only library routines $$$$$
c
c   Subroutine uctolc converts alphabetic characters in string ia from
c   upper case to lower case.  If ifl < 0 all characters are converted.
c   Otherwise characters enclosed by single quotes are left unchanged.
c   Programmed on 21 January by R. Buland.  Calling sequence changed
c   on 11 December 1985 by R. Buland.
c
      character*(*) ia
      data nfl/1/
      if(ifl.lt.0) nfl=1
c   Scan the string.
      n=len(ia)
      do 1 i=1,n
      if(ifl.lt.0) go to 2
c   Look for single quotes.
      if(ia(i:i).eq.'''') nfl=-nfl
c   If we are in a quoted string skip the conversion.
      if(nfl.lt.0) go to 1
c   Do the conversion.
 2    ib=ichar(ia(i:i))
      if(ib.lt.65.or.ib.gt.90) go to 1
      ia(i:i)=char(ib+32)
 1    continue
      return
      end
      subroutine r4sort(n,rkey,iptr)
c
c $$$$$ calls no other routine $$$$$
c
c   R4sort sorts the n elements of array rkey so that rkey(i), 
c   i = 1, 2, 3, ..., n are in asending order.  R4sort is a trivial
c   modification of ACM algorithm 347:  "An efficient algorithm for
c   sorting with minimal storage" by R. C. Singleton.  Array rkey is
c   sorted in place in order n*alog2(n) operations.  Coded on
c   8 March 1979 by R. Buland.  Modified to handle real*4 data on
c   27 September 1983 by R. Buland.
c
      save
      dimension rkey(n),iptr(n),il(10),iu(10)
c   Note:  il and iu implement a stack containing the upper and
c   lower limits of subsequences to be sorted independently.  A
c   depth of k allows for n<=2**(k+1)-1.
      if(n.le.0) return
      do 1 i=1,n
 1    iptr(i)=i
      if(n.le.1) return
      r=.375
      m=1
      i=1
      j=n
c
c   The first section interchanges low element i, middle element ij,
c   and high element j so they are in order.
c
 5    if(i.ge.j) go to 70
 10   k=i
c   Use a floating point modification, r, of Singleton's bisection
c   strategy (suggested by R. Peto in his verification of the
c   algorithm for the ACM).
      if(r.gt..58984375) go to 11
      r=r+.0390625
      go to 12
 11   r=r-.21875
 12   ij=i+(j-i)*r
      if(rkey(iptr(i)).le.rkey(iptr(ij))) go to 20
      it=iptr(ij)
      iptr(ij)=iptr(i)
      iptr(i)=it
 20   l=j
      if(rkey(iptr(j)).ge.rkey(iptr(ij))) go to 39
      it=iptr(ij)
      iptr(ij)=iptr(j)
      iptr(j)=it
      if(rkey(iptr(i)).le.rkey(iptr(ij))) go to 39
      it=iptr(ij)
      iptr(ij)=iptr(i)
      iptr(i)=it
 39   tmpkey=rkey(iptr(ij))
      go to 40
c
c   The second section continues this process.  K counts up from i and
c   l down from j.  Each time the k element is bigger than the ij
c   and the l element is less than the ij, then interchange the
c   k and l elements.  This continues until k and l meet.
c
 30   it=iptr(l)
      iptr(l)=iptr(k)
      iptr(k)=it
 40   l=l-1
      if(rkey(iptr(l)).gt.tmpkey) go to 40
 50   k=k+1
      if(rkey(iptr(k)).lt.tmpkey) go to 50
      if(k.le.l) go to 30
c
c   The third section considers the intervals i to l and k to j.  The
c   larger interval is saved on the stack (il and iu) and the smaller
c   is remapped into i and j for another shot at section one.
c
      if(l-i.le.j-k) go to 60
      il(m)=i
      iu(m)=l
      i=k
      m=m+1
      go to 80
 60   il(m)=k
      iu(m)=j
      j=l
      m=m+1
      go to 80
c
c   The fourth section pops elements off the stack (into i and j).  If
c   necessary control is transfered back to section one for more
c   interchange sorting.  If not we fall through to section five.  Note
c   that the algorighm exits when the stack is empty.
c
 70   m=m-1
      if(m.eq.0) return
      i=il(m)
      j=iu(m)
 80   if(j-i.ge.11) go to 10
      if(i.eq.1) go to 5
      i=i-1
c
c   The fifth section is the end game.  Final sorting is accomplished
c   (within each subsequence popped off the stack) by rippling out
c   of order elements down to their proper positions.
c
 90   i=i+1
      if(i.eq.j) go to 70
      if(rkey(iptr(i)).le.rkey(iptr(i+1))) go to 90
      k=i
      kk=k+1
      ib=iptr(kk)
 100  iptr(kk)=iptr(k)
      kk=k
      k=k-1
      if(rkey(ib).lt.rkey(iptr(k))) go to 100
      iptr(kk)=ib
      go to 90
      end
      function iupcor(phnm,dtdd,xcor,tcor)
      save
      include 'ttlim.inc'
      character*(*) phnm
      character*8 phcd
      double precision us,pt,tau,xlim,xbrn,dbrn,zs,pk,pu,pux,tauu,xu,
     1 px,xt,taut,coef,tauc,xc,tcoef,tp
      double precision x,dp,dps,ps,cn
      common/tabc/us(2),pt(jout),tau(4,jout),xlim(2,jout),xbrn(jbrn,3),
     1 dbrn(jbrn,2),xn,pn,tn,dn,hn,jndx(jbrn,2),idel(jbrn,3),mbr1,mbr2
      common/brkc/zs,pk(jseg),pu(jtsm0,2),pux(jxsm,2),tauu(jtsm,2),
     1 xu(jxsm,2),px(jbrn,2),xt(jbrn,2),taut(jout),coef(5,jout),
     2 tauc(jtsm),xc(jxsm),tcoef(5,jbrna,2),tp(jbrnu,2),odep,
     3 fcs(jseg,3),nin,nph0,int0(2),ki,msrc(2),isrc(2),nseg,nbrn,ku(2),
     4 km(2),nafl(jseg,3),indx(jseg,2),kndx(jseg,2),iidx(jseg),
     5 jidx(jbrn),kk(jseg)
      common/pcdc/phcd(jbrn)
      data oldep,jp,js/-1.,2*0/,cn/57.295779d0/
c
      iupcor=1
c     print *,'oldep odep',oldep,odep
      if(oldep.eq.odep) go to 1
      oldep=odep
c   Find the upgoing P branch.
c     print *,'mbr1 mbr2',mbr1,mbr2
      do 2 jp=mbr1,mbr2
c     print *,'jp phcd xbrn',jp,'  ',phcd(jp),xbrn(jp,1)
      if((phcd(jp).eq.'Pg'.or.phcd(jp).eq.'Pb'.or.phcd(jp).eq.'Pn'.or.
     1 phcd(jp).eq.'P').and.xbrn(jp,1).le.0d0) go to 3
 2    continue
      jp=0
c   Find the upgoing S branch.
 3    do 4 js=mbr1,mbr2
c     print *,'js phcd xbrn',js,'  ',phcd(js),xbrn(js,1)
      if((phcd(js).eq.'Sg'.or.phcd(js).eq.'Sb'.or.phcd(js).eq.'Sn'.or.
     1 phcd(js).eq.'S').and.xbrn(js,1).le.0d0) go to 1
 4    continue
      js=0
c
c1    print *,'jp js',jp,js
 1    if(phnm.ne.'P'.and.phnm.ne.'p') go to 5
      jb=jp
      if(jb)14,14,6
c
 5    if(phnm.ne.'S'.and.phnm.ne.'s') go to 13
      jb=js
      if(jb)14,14,6
c
 6    is=jndx(jb,1)+1
      ie=jndx(jb,2)
      ps=abs(dtdd)/dn
c     print *,'jb is ie dtdd dn ps',jb,is,ie,dtdd,dn,ps
      if(ps.lt.pt(is-1).or.ps.gt.pt(ie)) go to 13
      do 7 i=is,ie
c     print *,'i pt',i,pt(i)
      if(ps.le.pt(i)) go to 8
 7    continue
      go to 13
c
 8    j=i-1
      dp=pt(ie)-ps
      dps=dsqrt(dabs(dp))
      x=tau(2,j)+2d0*dp*tau(3,j)+1.5d0*dps*tau(4,j)
c     print *,'j pt dp dps x',j,pt(ie),dp,dps,x
      tcor=tn*(tau(1,j)+dp*(tau(2,j)+dp*tau(3,j)+dps*tau(4,j))+ps*x)
      xcor=cn*x
c     print *,'iupcor xcor tcor',iupcor,xcor,tcor
      return
c
 13   iupcor=-1
 14   xcor=0.
      tcor=0.
c     print *,'iupcor xcor tcor',iupcor,xcor,tcor
      return
      end
      subroutine brnset(nn,pcntl,prflg)
c
c   Brnset takes character array pcntl(nn) as a list of nn tokens to be
c   used to select desired generic branches.  Prflg(3) is the old
c   prnt(2) debug print flags in the first two elements plus a new print
c   flag which controls a branch selected summary from brnset.  Note that
c   the original two flags controlled a list of all tau interpolations
c   and a branch range summary respectively.  The original summary output
c   still goes to logical unit 10 (ttim1.lis) while the new output goes
c   to the standard output (so the caller can see what happened).  Each
c   token of pcntl may be either a generic branch name (e.g., P, PcP,
c   PKP, etc.) or a keyword (defined in the data statement for cmdcd
c   below) which translates to more than one generic branch names.  Note
c   that generic branch names and keywords may be mixed.  The keywords
c   'all' (for all branches) and 'query' (for an interactive token input
c   query mode) are also available.
c
      save
      parameter(ncmd=4,lcmd=16)
      include 'ttlim.inc'
      logical prflg(3),segmsk,prnt,fnd,all
      character*(*) pcntl(nn)
      character*8 phcd,segcd(jbrn),cmdcd(ncmd),cmdlst(lcmd),phtmp,
     1 phlst(jseg)
      double precision zs,pk,pu,pux,tauu,xu,px,xt,taut,coef,tauc,xc,
     1 tcoef,tp
      dimension nsgpt(jbrn),ncmpt(2,ncmd)
      common/brkc/zs,pk(jseg),pu(jtsm0,2),pux(jxsm,2),tauu(jtsm,2),
     1 xu(jxsm,2),px(jbrn,2),xt(jbrn,2),taut(jout),coef(5,jout),
     2 tauc(jtsm),xc(jxsm),tcoef(5,jbrna,2),tp(jbrnu,2),odep,
     3 fcs(jseg,3),nin,nph0,int0(2),ki,msrc(2),isrc(2),nseg,nbrn,ku(2),
     4 km(2),nafl(jseg,3),indx(jseg,2),kndx(jseg,2),iidx(jseg),
     5 jidx(jbrn),kk(jseg)
      common/pcdc/phcd(jbrn)
c   Segmsk is a logical array that actually implements the branch
c   editing in depset and depcor.
      common/prtflc/segmsk(jseg),prnt(2)
c
c   The keywords do the following:
c      P      gives P-up, P, Pdiff, PKP, and PKiKP
c      P+     gives P-up, P, Pdiff, PKP, PKiKP, PcP, pP, pPdiff, pPKP,
c             pPKiKP, sP, sPdiff, sPKP, and sPKiKP
c      S+     gives S-up, S, Sdiff, SKS, sS, sSdiff, sSKS, pS, pSdiff,
c             and pSKS
c      basic  gives P+ and S+ as well as ScP, SKP, PKKP, SKKP, PP, and
c             P'P'
c   Note that generic S gives S-up, Sdiff, and SKS already and so
c   doesn't require a keyword.
c
      data cmdcd/'P','P+','basic','S+'/
      data cmdlst/'P','PKiKP','PcP','pP','pPKiKP','sP','sPKiKP','ScP',
     1 'SKP','PKKP','SKKP','PP','S','ScS','sS','pS'/
      data ncmpt/1,2,1,7,1,13,13,16/
c
c   Take care of the print flags.
      prnt(1)=prflg(1)
      prnt(2)=prflg(2)
      if(prnt(1)) prnt(2)=.true.
c   Copy the token list into local storage.
      no=min0(nn,jseg)
      do 23 i=1,no
 23   phlst(i)=pcntl(i)
c   See if we are in query mode.
      if(no.gt.1.or.(phlst(1).ne.'query'.and.phlst(1).ne.'QUERY'))
     1 go to 1
c
c   In query mode, get the tokens interactively into local storage.
c
C 22   print *,'Enter desired branch control list at the prompts:'
      no=0
 21   call query(' ',fnd)
      if(no.ge.jseg) go to 1
      no=no+1
C      read 100,phlst(no)
      phlst(no) = 'all'
 100  format(a)
c   Terminate the list of tokens with a blank entry.
      if(phlst(no).ne.' ') go to 21
      no=no-1
      if(no.gt.0) go to 1
c   If the first token is blank, help the user out.
      print *,'You must enter some branch control information!'
      print *,'     possibilities are:'
      print *,'          all'
      print 101,cmdcd
 101  format(11x,a)
      print *,'          or any generic phase name'
C      go to 22
c
c   An 'all' keyword is easy as this is already the default.
 1    all=.false.
      if(no.eq.1.and.(phlst(1).eq.'all'.or.phlst(1).eq.'ALL'))
     1 all=.true.
      all=.true.
      if(all.and..not.prflg(3)) return
c
c   Make one or two generic branch names for each segment.  For example,
c   the P segment will have the names P and PKP, the PcP segment will
c   have the name PcP, etc.
c
      kseg=0
      j=0
c   Loop over the segments.
      do 2 i=1,nseg
      if(.not.all) segmsk(i)=.false.
c   For each segment, loop over associated branches.
 9    j=j+1
      phtmp=phcd(j)
c   Turn the specific branch name into a generic name by stripping out
c   the crustal branch and core phase branch identifiers.
      do 3 l=2,8
 6    if(phtmp(l:l).eq.' ') go to 4
      if(phtmp(l:l).ne.'g'.and.phtmp(l:l).ne.'b'.and.phtmp(l:l).ne.'n')
     1 go to 5
      if(l.lt.8) phtmp(l:)=phtmp(l+1:)
      if(l.ge.8) phtmp(l:)=' '
      go to 6
 5    if(l.ge.8) go to 3
      if(phtmp(l:l+1).ne.'ab'.and.phtmp(l:l+1).ne.'ac'.and.
     1 phtmp(l:l+1).ne.'df') go to 3
      phtmp(l:)=' '
      go to 4
 3    continue
c4    print *,'j phcd phtmp =',j,' ',phcd(j),' ',phtmp
c
c   Make sure generic names are unique within a segment.
 4    if(kseg.lt.1) go to 7
      if(phtmp.eq.segcd(kseg)) go to 8
 7    kseg=kseg+1
      segcd(kseg)=phtmp
      nsgpt(kseg)=i
c     if(prflg(3)) print *,'kseg nsgpt segcd =',kseg,nsgpt(kseg),' ',
c    1 segcd(kseg)
 8    if(jidx(j).lt.indx(i,2)) go to 9
 2    continue
      if(all) go to 24
c
c   Interpret the tokens in terms of the generic branch names.
c
      do 10 i=1,no
c   Try for a keyword first.
      do 11 j=1,ncmd
      if(phlst(i).eq.cmdcd(j)) go to 12
 11   continue
c
c   If the token isn't a keyword, see if it is a generic branch name.
      fnd=.false.
      do 14 k=1,kseg
      if(phlst(i).ne.segcd(k)) go to 14
      fnd=.true.
      l=nsgpt(k)
      segmsk(l)=.true.
c     print *,'Brnset:  phase found - i k l segcd =',i,k,l,' ',
c    1 segcd(k)
 14   continue
c   If no matching entry is found, warn the caller.
      if(.not.fnd) print *,'Brnset:  phase ',phlst(i),' not found.'
      go to 10
c
c   If the token is a keyword, find the matching generic branch names.
 12   j1=ncmpt(1,j)
      j2=ncmpt(2,j)
      do 15 j=j1,j2
      do 15 k=1,kseg
      if(cmdlst(j).ne.segcd(k)) go to 15
      l=nsgpt(k)
      segmsk(l)=.true.
c     print *,'Brnset:  cmdlst found - j k l segcd =',j,k,l,' ',
c    1 segcd(k)
 15   continue
 10   continue
c
c   Make the caller a list of the generic branch names selected.
c
 24   if(.not.prflg(3)) return
      fnd=.false.
      j2=0
c   Loop over segments.
      do 16 i=1,nseg
      if(.not.segmsk(i)) go to 16
c   If selected, find the associated generic branch names.
      j2=j2+1
      do 17 j1=j2,kseg
      if(nsgpt(j1).eq.i) go to 18
 17   continue
      print *,'Brnset:  Segment pointer (',i,') missing?'
      go to 16
 18   do 19 j2=j1,kseg
      if(nsgpt(j2).ne.i) go to 20
 19   continue
      j2=kseg+1
c   Print the result.
 20   j2=j2-1
C     if(.not.fnd) print *,'Brnset:  the following phases have '//
C    1 'been selected -'
C     fnd=.true.
C     print 102,i,(segcd(j),j=j1,j2)
 102  format(10x,i5,5(2x,a))
 16   continue
      return
      end




      logical function oneray(phnm,dtdd,xcor,tcor)
c
c   Given a phase code, phnm, oneray returns the distance, xcor, in 
c   degrees and the travel time, tcor, in seconds corresponding to ray 
c   parameter dtdd in km/(km*s).
c
      save
      double precision cn
      parameter(cn=57.295779d0)
      include 'ttlim.inc'
      character*(*) phnm
      character*8 phcd,phsyn
      double precision us,pt,tau,xlim,xbrn,dbrn,zs,pk,pu,pux,tauu,xu,
     1 px,xt,taut,coef,tauc,xc,tcoef,tp
      double precision x,dp,dps,ps
      common/tabc/us(2),pt(jout),tau(4,jout),xlim(2,jout),xbrn(jbrn,3),
     1 dbrn(jbrn,2),xn,pn,tn,dn,hn,jndx(jbrn,2),idel(jbrn,3),mbr1,mbr2
      common/brkc/zs,pk(jseg),pu(jtsm0,2),pux(jxsm,2),tauu(jtsm,2),
     1 xu(jxsm,2),px(jbrn,2),xt(jbrn,2),taut(jout),coef(5,jout),
     2 tauc(jtsm),xc(jxsm),tcoef(5,jbrna,2),tp(jbrnu,2),odep,
     3 fcs(jseg,3),nin,nph0,int0(2),ki,msrc(2),isrc(2),nseg,nbrn,ku(2),
     4 km(2),nafl(jseg,3),indx(jseg,2),kndx(jseg,2),iidx(jseg),
     5 jidx(jbrn),kk(jseg)
      common/pcdc/phcd(jbrn)
c
      oneray=.true.
      phsyn=phnm
      j=index(phsyn,'bc')
      if(j.gt.0) phsyn(j:j+1)='ab'
c
c   Find the branch.
c     print *,'mbr1 mbr2',mbr1,mbr2
      do jb=mbr1,mbr2
c       print *,'jb phcd xbrn',jb,'  ',phcd(jb),xbrn(jb,1)
        if(phcd(jb).eq.phsyn) then
c
c   Got the branch.  See if the ray parameter is OK.
          is=jndx(jb,1)+1
          ie=jndx(jb,2)
          ps=abs(dtdd)/dn
c         print *,'jb is ie dtdd dn ps',jb,is,ie,dtdd,dn,ps
          if(ps.ge.pt(is-1).and.ps.le.pt(ie)) then
c
c   The ray parameter is OK.  Find the right ray parameter interval.
            do i=is,ie
c             print *,'i pt',i,pt(i)
c
c   Got the ray parameter interval.  Interpolate.
              if(ps.le.pt(i)) then
                j=i-1
                dp=pt(ie)-ps
                dps=dsqrt(dabs(dp))
                x=tau(2,j)+2d0*dp*tau(3,j)+1.5d0*dps*tau(4,j)
c               print *,'j pt dp dps x',j,pt(ie),dp,dps,x
                tcor=tn*(tau(1,j)+dp*(tau(2,j)+dp*tau(3,j)+
     1           dps*tau(4,j))+ps*x)
                xcor=cn*x
c               print *,'oneray xcor tcor',oneray,xcor,tcor
                return
              endif
            enddo
          endif
        endif
      enddo
c
c   Didn't get it.
      oneray=.false.
      xcor=0.
      tcor=0.
c     print *,'oneray xcor tcor',oneray,xcor,tcor
      return
      end
c $$$
c  The following routines, warn, tnoua, and dasign, were is a separate file named libsun.a
c  The all seem to work on all platforms in g77, so I have simply added them to libtau.f
c  jas/vt October 2007
c
      subroutine warn(msg)
      character*(*) msg
      write(*,100) msg
 100  format(1x,a)
      return
      end
      subroutine tnoua(ia,n)
c
c $$$$$ calls no other routine $$$$$
c
c   Subroutine tnoua writes the first n characters of string ia to the
c   standard output without the trailing newline (allowing user input
c   on the same line).  Programmed on 17 September 1980 by
c   R. Buland.
c
      character*(*) ia
C      write(*,100)ia(1:n)
 100  format(a,$)
      return
      end
      subroutine dasign(lu,mode,ia,len)
c
c $$$$$ calls no other routine $$$$$
c
c   Subroutine dasign opens (connects) logical unit lu to the disk file
c   named by the character string ia with mode mode.  If iabs(mode) = 1,
c   then open the file for reading.  If iabs(mode) = 2, then open the
c   file for writing.  If iabs(mode) = 3, then open a scratch file for
c   writing.  If mode > 0, then the file is formatted.  If mode < 0,
c   then the file is unformatted.  All files opened by dasign are
c   assumed to be direct access.  Programmed on 3 December 1979 by
c   R. Buland.
c
      save
      character*(*) ia
      logical exst
c
      if(mode.ge.0) nf=1
      if(mode.lt.0) nf=2
      ns=iabs(mode)
      if(ns.le.0.or.ns.gt.3) ns=3
      go to (1,2),nf
 1    go to (11,12,13),ns
 11   open(lu,file=ia,status='old',form='formatted',
     1 access='direct',recl=len)
      return
 12   inquire(file=ia,exist=exst)
      if(exst) go to 11
 13   open(lu,file=ia,status='new',form='formatted',
     1 access='direct',recl=len)
      return
 2    go to (21,22,23),ns
 21   open(lu,file=ia,status='old',form='unformatted',access='direct',
     1 recl=len)
      return
 22   inquire(file=ia,exist=exst)
      if(exst) go to 21
 23   open(lu,file=ia,status='new',form='unformatted',access='direct',
     1 recl=len)
      return
      end
      subroutine vexit(ierr)
      call exit(ierr)
      end
C+
	function lenc(string)
C
C	Returns length of character variable STRING excluding right-hand
C	  most blanks or nulls
C-
	character*(*) string
	length = len(string)	! total length
	if (length .eq. 0) then
	  lenc = 0
	  return
	end if
	if(ichar(string(length:length)).eq.0)string(length:length) = ' '
	do j=length,1,-1
	  lenc = j
	  if (string(j:j).ne.' ' .and. ichar(string(j:j)).ne.0) return
	end do
	lenc = 0
	return
	end
