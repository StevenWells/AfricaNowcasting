subroutine lst_anom(iyr,imon,iday,ihr,imin,mount_drive)

! INPUT
! iyr,imon, iday, ihr, imin (int): date to caluclate anomaly for

! mount_drive (int): to fix root path to files, prefix with mnt if mount_drive==1
! outDir (character): output directory for anomaly file          


  use netcdf
  implicit none

   ! mount_drive
   integer :: mount_drive
   character*4::mnt

   ! set up to do a particular day
   integer ::  iyr,imon,iday, ihr,imin, ip



   ! output directory
   character*300 ::  outDir
   character*300 ::  ancilDir


  ! take MTG LST image and create anomaly wrt existing MSG LST climatology

  real,parameter::del=0.03,d2=del/2., &
       lon1=-17.5+d2,lon2=52.5-d2,lat1=-36+d2,lat2=38.-d2, &
       ltime_min=9., ltime_max=17.    ! window in local time used to compute daytime mean lsta
  
  ! THIS DOES A SINGLE DAY - date is variable so declared separately
  integer,parameter::onx=(nint(lon2-lon1)/del)+1,ony=nint((lat2-lat1)/del)+1
  integer :: year1, year2, mon1, mon2, ndays


  ! THIS LOOPS OVER DATES.
  !   integer,parameter::onx=(nint(lon2-lon1)/del)+1,ony=nint((lat2-lat1)/del)+1,&
  !     year1=2025,year2=2026,mon1=1+1,mon2=12,ndays=31
  
  integer::status,ncid,varid,ndims,nvars,nglobalatts,unlimdimid,len,numdims
  character*200::fname,name,chvar
  integer,parameter::nx3=5568,ny3=5568,&
       !       nph=6,nph2=4,hour1=4,hour2=18, &
       nph=6,nph2=4, &
       hour1=floor(ltime_min-52.5/15.),hour2=floor(ltime_max+17.5/15.), &
       nslots=(hour2-hour1+1)*nph,nslots2=(hour2-hour1+1)*nph2+1, &
       nx2=2326,ny2=2599,&
       nwin=6, &
       nclim_min=50                       !from Seonaid's code

  logical,parameter::lcompute_weights=.false.

  real,parameter::undef=-999.9,min_wt=0.99  !minimum weighting for interpolation

  integer::i,j,ii,jj,i3,j3,hour,mins,slot,s1,s2,year,mon,day, &
       last_i=1,last_j=1,i1=nx3,j1=ny3,i2=0,j2=0

  real,dimension(nx3,ny3)::lonm3,latm3,lst2_mtg
  real,allocatable,dimension(:,:,:)::lst3,clim2_mtg,dlst_mtg,clim15,clim2,lsta_mtg
  real,dimension(nx2,ny2)::lonm2,latm2,lst2

!  real::clim15(nx2,ny2,nslots2),clim2(nx2,ny2,nslots)
  integer(kind=8),dimension(nx2,ny2)::nclim
  real(kind=8),dimension(nx2,ny2)::din

  real,dimension(onx,ony)::n=0.,lst=0.
  integer(kind=2)::out(onx,ony)
  integer(kind=2)::ilst(nx3,ny3),ij_nw(nx3,ny3,2)=0,iwt(nx3,ny3,2,2)
  integer(kind=2),parameter::i2miss=-29999,iundef=-9999

  logical::ll=.false.
  real::lon3,lat3,wt(nx3,ny3,2,2),msg_lst_on_mtg_grid,wts2,ltime,swt(nslots)
  character*2::cmon,cday
  character*4::ctime,cltime,cyear

  allocate(clim15(nx2,ny2,nslots2),clim2(nx2,ny2,nslots))


   year1=iyr
   year2=iyr
   mon1=imon
   mon2=imon
   ndays=1
   
   
   if (mount_drive.eq.1) then
      mnt = '/mnt'
   else
      mnt = ''
   end if
   ancilDir = '/home/stewells/AfricaNowcasting/ancils/mtg_lst/'
   outDir = mnt//'/prj/swift/MTG_LST/Africa/'
  if(nx3.eq.5568) then
     call read_mtg_grid_full(lonm3,latm3,undef,mnt)
  else
     stop
  endif

  call read_msg_ssa_lat_lon(nx2,ny2,lonm2,latm2,undef,mnt)

  print*,minval(lonm3,lonm3.ne.undef),maxval(lonm3),minval(latm3,latm3.ne.undef),maxval(latm3)
  print*,minval(lonm2,lonm2.ne.undef),maxval(lonm2),minval(latm2,latm2.ne.undef),maxval(latm2)

  wt = undef

  if(lcompute_weights) then

     do j=1,ny3
        do i=1,nx3
        
           if(i.eq.1) last_i = 1
        
           if(lonm3(i,j).lt.lon1 .or. lonm3(i,j).gt.lon2) cycle
           if(latm3(i,j).lt.lat1 .or. latm3(i,j).gt.lat2) cycle

           lon3 = lonm3(i,j) ; lat3 = latm3(i,j)

           jj_loop : do jj=last_j,ny2-1

              if(last_j.gt.1 .and. jj.gt.last_j+nwin) then
                 print*,'gone beyond window,',jj,last_j
                 print*,i,j,lon3,lat3,last_i,last_j
                 print*,lonm2(last_i,last_j),latm2(last_i,last_j)
                 stop
              endif

              ii_loop : do ii=last_i,nx2-1
              
                 if(lonm2(ii,jj).lt.lon3.and.lonm2(ii+1,jj).ge.lon3.and. &
                      latm2(ii,jj).gt.lat3.and.latm2(ii,jj+1).le.lat3) then

                    ij_nw(i,j,1) = ii ; ij_nw(i,j,2) = jj

                    call MSG_weights(lon3,lat3, &
                         lonm2(ii:ii+1,jj:jj+1), &
                         latm2(ii:ii+1,jj:jj+1), wt(i,j,:,:))

                    last_i = ii - 1 ; last_j = jj - 1

                    if(i.lt.i1) i1=i ; if(i.gt.i2) i2=i
                    if(j.lt.j1) j1=j ; if(j.gt.j2) j2=j
                    exit jj_loop

                 endif

                 if(lonm2(ii,jj).gt.lon3) exit ii_loop              

              enddo ii_loop
           enddo jj_loop
        enddo
     enddo

     allocate(lst3(i1:i2,j1:j2,nslots),clim2_mtg(i1:i2,j1:j2,nslots), &
          dlst_mtg(i1:i2,j1:j2,nslots),lsta_mtg(i1:i2,j1:j2,0:1))
     print*,i1,i2,i2-i1+1,j1,j2,ny3-j2+1,ny3-j1+1,j2-j1+1
     
     iwt = nint(1000.*wt)
     !open(1,file=mnt//'/scratch/cmt/mtg2msg.gra',form='unformatted',access='direct',&
     open(1,file=ancilDir//'/mtg2msg.gra',form='unformatted',access='direct',&
          recl=2*nx3*ny3)
     write(1,rec=1) ij_nw(:,:,1) ;  write(1,rec=2) ij_nw(:,:,2) ; close(1)
     
     close(1)
     
     ! open(1,file= mnt //'/prj/swift/MTG_LST/Africa/MSG_2326_2599_MTG_3300_3670_interpolation.gra', &
     open(1,file= ancilDir//'/MSG_2326_2599_MTG_3300_3670_interpolation.gra', &
          form='unformatted',access='direct',recl=2*(i2-i1+1)*(j2-j1+1))
     write(1,rec=1) ij_nw(i1:i2,j1:j2,1)
     write(1,rec=2) ij_nw(i1:i2,j1:j2,2)

     do jj=1,2
     do ii=1,2
        write(1,rec=2+(jj-1)*2+ii) iwt(i1:i2,j1:j2,ii,jj)
     enddo
     enddo
     close(1)
   
     !open(1,file=mnt //'/prj/swift/MTG_LST/Africa/lon_lat_3300_3670.gra',form='unformatted', &
     open(1,file= ancilDir//'/lon_lat_3300_3670.gra',form='unformatted', &
          access='direct',recl=4*(i2-i1+1)*(j2-j1+1))
     write(1,rec=1) lonm3(i1:i2,j1:j2)
     write(1,rec=2) latm3(i1:i2,j1:j2)
     close(1)
  else
     
     print*,'reading pre-existing interpolation data for MSG to MTG'
     i1 = 1836 ; i2 = i1 + 3300 - 1
     j1 = 911  ; j2 = j1 + 3670 - 1
     allocate(lst3(i1:i2,j1:j2,nslots),clim2_mtg(i1:i2,j1:j2,nslots), &
          dlst_mtg(i1:i2,j1:j2,nslots),lsta_mtg(i1:i2,j1:j2,0:1))
     
     !open(1,file=mnt //'/prj/swift/MTG_LST/Africa/MSG_2326_2599_MTG_3300_3670_interpolation.gra', &
     fname = mnt//'/prj/swift/MTG_LST/Africa/MSG_2326_2599_MTG_3300_3670_interpolation.gra'
     open(1,file=fname, &
          form='unformatted',access='direct',recl=2*(i2-i1+1)*(j2-j1+1),status='old')
     read(1,rec=1) ij_nw(i1:i2,j1:j2,1)
     read(1,rec=2) ij_nw(i1:i2,j1:j2,2)
     do jj=1,2
     do ii=1,2
        read(1,rec=2+(jj-1)*2+ii) iwt(i1:i2,j1:j2,ii,jj)
     enddo
     enddo
     close(1)
     wt = iwt/1000.
     
  endif
  !print*,maxval(wt) ; stop

  do year=year1,year2
  do mon=mon1,mon2
  do day=iday,iday      
   print *, year,mon,day
  ! read in full day of MSG clim data and interpolate onto MTG time step

  do slot=1,nslots2
     hour = floor((slot-1.)/nph2) + hour1
     mins  = mod(slot-1,nph2)*15
     write(ctime,'(i4.4)') hour*100+mins

     write(cyear,'(i4)') year ; write(cmon,'(i2.2)') mon ; write(cday,'(i2.2)') day
     
! read climatology file on MSG grid

     fname = mnt //'/prj/swift/SEVIRI_LST/SEVIRI_LST_2004-2022/historic_clim/'//cmon// &
          '/HDF5_LSASAF_MSG_LST_MSG-Disk_HistStats_'//cmon//cday//'_'//ctime

     status = nf90_open(path=trim(fname),mode=nf90_nowrite,ncid=ncid)
     if(status.ne.0) cycle
     print*,slot,status,trim(fname)
  
     status=nf90_inq_varid(ncid,'Mean',varid)
     if(status.ne.0) print*,nf90_strerror(status)
     status=nf90_get_var(ncid,varid,din)
     if(status.ne.0) print*,nf90_strerror(status)

     where(isnan(din))
        clim15(:,:,slot) = undef
     elsewhere
        clim15(:,:,slot) = din/100.
     endwhere
  
     status=nf90_inq_varid(ncid,'count',varid)
     if(status.ne.0) print*,nf90_strerror(status)
     status=nf90_get_var(ncid,varid,nclim)
     if(status.ne.0) print*,nf90_strerror(status)
     
     where(nclim.lt.nclim_min) clim15(:,:,slot) = undef

     status = nf90_close(ncid)
  enddo

! interpolate climatological MSG data every 15 minutes (clim15) onto MTG time step (clim2)
  
  do slot=1,nslots
     s1 = floor((slot-1)*real(nph2)/nph) + 1
     s2 = ceiling((slot-1)*real(nph2)/nph) + 1
     wts2 = ( (slot-1.)/nph - (s1-1.)/nph2 ) * nph2
     where(clim15(:,:,s1).ne.undef) clim2(:,:,slot) = (1-wts2)*clim15(:,:,s1) + wts2*clim15(:,:,s2)
     where(clim15(:,:,s2).eq.undef) clim2(:,:,slot) = undef
  enddo

! interpolate MSG climatology on to MTG grid for all slots in day

  clim2_mtg = 0.
  print*,'putting MSG clim on MTG grid for all slots'

  do j=j1,j2
     if(mod(j,250).eq.0) print*,j
  do i=i1,i2

     if(ij_nw(i,j,1).eq.0) then
        clim2_mtg(i,j,:) = undef
        cycle
     endif

     swt = 0.

     do jj=ij_nw(i,j,2),ij_nw(i,j,2)+1
     do ii=ij_nw(i,j,1),ij_nw(i,j,1)+1

        i3=ii-ij_nw(i,j,1)+1 ; j3 = jj-ij_nw(i,j,2)+1
        where(clim2(ii,jj,:).ne.undef)
           clim2_mtg(i,j,:) = clim2_mtg(i,j,:) + clim2(ii,jj,:) * wt(i,j,i3,j3)
           swt = swt + wt(i,j,i3,j3)
        endwhere
       
     enddo
     enddo

     where(swt.ge.min_wt)
        clim2_mtg(i,j,:) = clim2_mtg(i,j,:)/swt
     elsewhere
        clim2_mtg(i,j,:) = undef
     endwhere
  
  enddo
  enddo

  print*,'done putting MSG on MTG grid'
  
  dlst_mtg = undef ; 
  
! loop over all MTG slots in day
  
  do slot=1,nslots
     hour = floor((slot-1.)/nph) + hour1
     mins  = mod(slot-1,nph)*10
     write(ctime,'(i4.4)') hour*100+mins
     ! LATEST FILE HERE
! read MTG LST file  
     fname=mnt//'/scratch/stewells/MTG_LST/full_disc/PRODUCTS/MTG/MTLST/NATIVE/'// &
           cyear//'/'//cmon//'/'//cday//'/LSA-007_MTG_MTLST_MTG-FD_'// &
          cyear//cmon//cday//ctime//'.nc'
     !fname=mnt//'/prj/swift/MTG_LST/full_disc/'//cyear//'/'//cmon//'/LSA-007_MTG_MTLST_MTG-FD_'// &
      !    cyear//cmon//cday//ctime//'.nc'
     status = nf90_open(path=trim(fname),mode=nf90_nowrite,ncid=ncid)
     print*,status,trim(fname)
     if(status.eq.0) then
        status=nf90_inq_varid(ncid,'LST',varid)
        if(status.ne.0) print*,nf90_strerror(status)
        status=nf90_get_var(ncid,varid,ilst)
        if(status.ne.0) print*,nf90_strerror(status)
        lst3(:,:,slot) = ilst(i1:i2,j1:j2)/100.
        where(lst3(:,:,slot).eq.i2miss/100.) lst3(:,:,slot) = undef
     else
        lst3(:,:,slot) = undef
     endif
     
     status = nf90_close(ncid)

! compute MTG anomaly relative to (interpolated) MSG climatology


     do j=j1,j2
        do i=i1,i2
           if(lst3(i,j,slot).ne.undef.and.clim2_mtg(i,j,slot).ne.undef) &
                dlst_mtg(i,j,slot) = lst3(i,j,slot) - clim2_mtg(i,j,slot)
        enddo
     enddo

!        open(1,file=mnt//'/scratch/cmt/mtg_test2_lst_'//cmon//cday//ctime//'.gra', &
!             form='unformatted',access='direct',recl=4*(i2-i1+1)*(j2-j1+1))
!        write(1,rec=1)lst3(:,:,slot) ; write(1,rec=2) dlst_mtg(:,:,slot) ;write(1,rec=3) clim2_mtg(:,:,slot)
!        close(1)
        
  enddo

  lsta_mtg = 0.
  print*,'creating mean lsta between ',ltime_min,' and ',ltime_max
  
  do slot=1,nslots
     hour = floor((slot-1.)/nph) + hour1
     mins  = mod(slot-1,nph)*10
     s1 = max(slot-2,1) ; s2 = min(slot+1,nslots) 
     do j=j1,j2
        do i=i1,i2
           ltime = hour + mins/60. + lonm3(i,j)/15.
           
! exclude slots where there is cloud in preceding 2 slots or following 1 slot           

           if ( minval(dlst_mtg(i,j,s1:s2)) .eq. undef) cycle
           if(i.eq.1800.and.j.eq.i) print*,slot,ltime,dlst_mtg(i,j,s1:s2)
           if(ltime.ge.ltime_min.and.ltime.le.ltime_max) then
              lsta_mtg(i,j,0) = lsta_mtg(i,j,0) + 1.
              lsta_mtg(i,j,1) = lsta_mtg(i,j,1) + dlst_mtg(i,j,slot)
           endif
        enddo
     enddo
  enddo

   fname = trim(outDir) // cyear// '/' //cmon// '/' //cday // '/' // &
   'mtg_lsta_0917_'//cyear//cmon//cday//'.gra'
   print *, fname
  where(lsta_mtg(:,:,0).eq.0.)
     lsta_mtg(:,:,1) = undef
  elsewhere
     lsta_mtg(:,:,1) = lsta_mtg(:,:,1)/lsta_mtg(:,:,0)
  endwhere

  
 ! open(1,file=mnt//'/scratch/cmt/mtg_lsta_0917_'//cyear//cmon//cday//'.gra', &
  open(1,file=fname, &
       form='unformatted',access='direct',recl=4*(i2-i1+1)*(j2-j1+1)*2)
  write(1,rec=1) lsta_mtg ; close(1)

  enddo
  enddo
  enddo           

end subroutine lst_anom


subroutine read_mtg_grid_full(lon,lat,undef,mnt)
  use netcdf  
  implicit none

  ! reads in netcdf file of locations on full grid

!  integer::nx,ny
  integer,parameter::nx=5568,ny=5568
  real,dimension(nx,ny)::lon,lat
  integer,dimension(nx,ny)::ilon,ilat
  integer,parameter::imiss=-910000
  integer::status,ncid,varid,ndims,nvars,nglobalatts,unlimdimid,len,numdims
  character*200::fname,name
  character*4::mnt
  real::undef
  
  fname=mnt//'/prj/swift/MTG_LST/full_disc/LSA_MTG_LATLON_MTG-FD_202312040900.nc'
  !fname=mnt//'/prj/swift/MTG_LST/full_disc/'//syr//'/'//smon//'/LSA-007_MTG_LATLON_MTG-FD_'//syr//smon//sday//shr//smin//'.nc'

  status = nf90_open(path=trim(fname),mode=nf90_nowrite,ncid=ncid)
    if(status.ne.0) then
       print *, "ERROR: ", fname, nf90_strerror(status)
    end if

  print*,status,trim(fname)
  status=nf90_inq_varid(ncid,'LON',varid)
  if(status.ne.0) print*,nf90_strerror(status)
  status=nf90_get_var(ncid,varid,ilon)
  if(status.ne.0) print*,nf90_strerror(status)
  status=nf90_inq_varid(ncid,'LAT',varid)
  if(status.ne.0) print*,nf90_strerror(status)
  status=nf90_get_var(ncid,varid,ilat)
  if(status.ne.0) print*,nf90_strerror(status)


  lon=undef ; lat=undef 
  
  where(ilon.ne.imiss)
     lon = ilon/10000.
     lat = ilat/10000.
  endwhere

  return
  end
!
!----------------------------------------------------------------------------
!
subroutine read_msg_ssa_lat_lon(nx,ny,lon,lat,undef,mnt)
use netcdf
implicit none
!
! reads in coordinates of Pan-African lsta grid (nx=2326,ny=2599)
!
integer::nx,ny,k
real::lon(nx,ny),lat(nx,ny),undef
integer::status,ncid,varid,ndims,nvars,nglobalatts,unlimdimid,len,numdims
character*200::fname,name
character*4::mnt
integer,parameter::i0=719,j0=1300

logical::lcheck_MSG_sub=.false.
integer,parameter::nx_full=3712,ny_full=3712, &
     i1_full=nx_full-2574+1,j1_full=557  !found from running ~/lsta_SSA/locate_lsta_PanAfrica.f90
real(kind=4),dimension(nx_full,ny_full)::lon_full,lat_full

fname=mnt//'/prj/swift/SEVIRI_LST/Ancillary/lsasaf_PanAfrica_latlon.nc'
!
status = nf90_open(path=trim(fname),mode=nf90_nowrite,ncid=ncid)
    if(status.ne.0) then
       print *, "ERROR: ", fname, nf90_strerror(status)
    end if
print*,status,trim(fname)
!
status=nf90_inquire(ncid,ndims,nvars,nglobalatts,unlimdimid)
!
do k=1,nvars
  status=nf90_inquire_variable(ncid,k,name,ndims=numdims)
  print*,k,trim(name),ndims
enddo
!
status=nf90_inq_varid(ncid,'latitude',varid)
if(status.ne.0) print*,nf90_strerror(status)
status=nf90_get_var(ncid,varid,lat)
if(status.ne.0) print*,nf90_strerror(status)
print*,'grid latitude range:',minval(lat),maxval(lat),lat(nx/2,1),lat(nx/2,ny)
!
status=nf90_inq_varid(ncid,'longitude',varid)
if(status.ne.0) print*,nf90_strerror(status)
status=nf90_get_var(ncid,varid,lon)
if(status.ne.0) print*,nf90_strerror(status)
print*,'grid longitude range:',minval(lon),maxval(lon),lon(1,ny/2),lon(nx,ny/2)
status=nf90_close(ncid)
print*,'location of (1,1) on pan-African grid',lon(1,1),lat(1,1)
!
! some pixels near edge of image have undefined coordinates which have been
! stored as 0,0. Set these to undef
!
where(lon.eq.0.and.lat.eq.0.) 
  lon=undef ; lat=undef
endwhere
lon(i0,j0) = 0. ; lat(i0,j0) = 0.  ! fix problem at 0,0 introduced by above line
!
if(lcheck_MSG_sub) then
   print*,'checking indices of (1,1) pixel on pan-African grid'
   fname = mnt//'/prj/swift/SEVIRI_LST/Ancillary/MSG_000_LatLon.nc'
   status = nf90_open(path=trim(fname),mode=nf90_nowrite,ncid=ncid)
   print*,status,trim(fname)
   if(status.ne.0) then ; print*,nf90_strerror(status) ; stop ; endif
 
   status=nf90_inq_varid(ncid,'lon',varid)
   if(status.ne.0) print*,nf90_strerror(status)
   status=nf90_get_var(ncid,varid,lon_full)
   if(status.ne.0) print*,nf90_strerror(status)
   status=nf90_inq_varid(ncid,'lat',varid)
   if(status.ne.0) print*,nf90_strerror(status)
   status=nf90_get_var(ncid,varid,lat_full)
   if(status.ne.0) print*,nf90_strerror(status)
   print*,i1_full,j1_full,lon_full(i1_full,j1_full),lat_full(i1_full,j1_full)
   stop
endif

return
end subroutine read_msg_ssa_lat_lon


!
!========================================================================
!
subroutine MSG_weights(lon,lat,mlon,mlat,wt)
  implicit none
!
! returns weights of 4 surrounding MSG pixels for given MTG pixel at lon,lat  
!
! location of MSG pixels in 2x2 array relative to MTG pixel at X
!
!   (1,1)     (2,1)
!
!         X
!
!   (1,2)     (2,2)
!

  real::lon,lat,mlon(2,2),mlat(2,2),wt(2,2),lon1,lon2,lat1,lat2,dlon,dlat

  
  lon1 = 0.5 * (mlon(1,1) + mlon(1,2))
  lon2 = 0.5 * (mlon(2,1) + mlon(2,2))

  lat1 = 0.5 * (mlat(1,1) + mlat(2,1))
  lat2 = 0.5 * (mlat(1,2) + mlat(2,2))

  dlon = lon2 - lon1
  dlat = lat2 - lat1

  wt(1,1) = (1. - (lat-lat1)/dlat) * (1. - (lon-lon1)/dlon)
  wt(1,2) = (1. - (lat2-lat)/dlat) * (1. - (lon-lon1)/dlon)
  wt(2,1) = (1. - (lat-lat1)/dlat) * (1. - (lon2-lon)/dlon)
  wt(2,2) = (1. - (lat2-lat)/dlat) * (1. - (lon2-lon)/dlon)
  


return
end subroutine MSG_weights


!--------------------------------------------------------------------------------------

subroutine read_MSG_LST_full_disk(nx,ny,lst,undef)
  implicit none

  ! reads binary file created by test_read_hdf.py containing full disk MSG lst
  ! and returns lst on SSA domain (nx,ny)

  integer::nx,ny
  integer,parameter::nx_full=3712,ny_full=3712, &
       i1_full=nx_full-2574+1,j1_full=557  !found from running ~/lsta_SSA/locate_lsta_PanAfrica.f90
  real::undef,lst(nx,ny),lst_full(nx_full,ny_full)

  open(1,file='output.bin',status='old',form='unformatted',access='direct',recl=4*nx_full*ny_full)
  read(1,rec=1) lst_full ; close(1)
  where(isnan(lst_full)) lst_full = undef

  print*,'MSG LST over full disk',minval(lst_full,lst_full.ne.undef),maxval(lst_full)
  lst = lst_full(i1_full:i1_full+nx-1,j1_full:j1_full+ny-1)
  print*,'MSG LST over pan-African domain',minval(lst,lst.ne.undef),maxval(lst)

  return
end subroutine read_MSG_LST_full_disk

!--------------------------------------------------------------------------------------

function msg_lst_on_mtg_grid(lst_msg,wt,min_wt,undef)
  implicit none

! THIS FUNCTION NO LONGER USED
  
  integer,parameter::n=4
  real,dimension(n)::lst_msg,wt
  integer::i
  real::msg_lst_on_mtg_grid,undef,swt,min_wt

  swt=0. ; msg_lst_on_mtg_grid = 0.
  do i=1,n
     if(lst_msg(i).ne.undef) then
        msg_lst_on_mtg_grid = msg_lst_on_mtg_grid + lst_msg(i) * wt(i)
        swt = swt + wt(i)
      endif
  enddo
  
  if(swt.ge.min_wt) then
     msg_lst_on_mtg_grid = msg_lst_on_mtg_grid / swt
  else
     msg_lst_on_mtg_grid = undef
  endif
  
  return
end function msg_lst_on_mtg_grid

     
