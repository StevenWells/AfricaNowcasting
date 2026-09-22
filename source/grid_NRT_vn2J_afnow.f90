program grid_NRT_vn2J

  use netcdf
  implicit none
  
  ! code to read individual netcdf files downloaded from HSAF on given day
  ! then create gridded sm data at 0.1deg
  ! and use pre-existing gridded climatology to create daily anomaly files
  ! using filters from ~/ASCAT/H129_TUW/nc_grid_ascat_H129_vn2X.f90

  ! This version differs from vn2H (used in Nature paper and in NRT till
  ! Aug 2026) as it no longer filters out edge pixels, which were
  ! a problem in earlier versions of the algorithm

  !! 18/3/25
  !! filter out edge pixels which were creating wet discontinuities
  !! based on updated version of  ~/ASCAT/H129_TUW/nc_grid_ascat_H129.f90
  !! this may need to be removed at some point when Sebastian implements an
  !! operational fix

  ! 25/3/25
  ! increase ssp_max to 20% from 10% to reduce filtering of
  ! semi-arid/arid regions

  ! 27/3/25
  ! now using vn2c climatology as well as above two changes

  ! 3/4/25
  ! v2H switches off ssp_max filter and instead uses sub-scattering from
  ! a TUW dataset of subsurface scattering signal strength

  ! 19/9/26
  ! SCW
  ! Made copy to be used on Africa Nowcasting machine
  ! Same processing, different in/out locations
  ! gfortran grid_NRT_vn2J_afnow.f90 $(nf-config --fflags) $(nf-config --flibs) -o grid_NRT_vn2J_afnow
  
  real,parameter::lon1=-17.95,lon2=54.95,lat1=-35.05,lat2=25.05,del=0.1
  integer,parameter::nx=nint((lon2-lon1)/del)+1,ny=nint((lat2-lat1)/del)+1, &
                     nbig=1000000,year0=1970,year2=2040,max_days=366,npass=2
  integer::nf=1,f,k,iop,i,j,l,yy,mm,dd,y,m,d
  character*200::fname,fpath,dim_name
  integer::status,ncid,varid,ndims,nvars,nglobalatts,unlimdimid,len,numdims
  integer(kind=4),dimension(nbig)::id,ilon,ilat,ssms
  integer(kind=1),dimension(nbig)::conf_flag,ipass,ssp,tc,pflag,sflag
  real(kind=8),dimension(nbig)::time
  integer(kind=2),dimension(nbig)::sm_in,ssmn
  real,dimension(nx,ny,npass)::smtot=0.,ntot=0.,sm_clim
  integer(kind=2)::daily(nx,ny,npass,2),anom(nx,ny,npass),ssub(nx,ny)
  
  
  logical::od
  character*3::ampm
  character*8::datestamp
  character*4::ystr
  character*2::mstr,dstr
  real::utc_lt,lt
  integer,dimension((year2-year0+1)*max_days)::year,mon,day
  
  real,parameter::lt_eq=9.5,dt_min=2.-0.5,&   !equatorial overpass time and 
                                        !permitted tolerance (hours) to use data
                ssms_min=1.E7, &        !min sensitivity 1dB * scale factor
                ssp_max=100.,  &        !100% switches off filter
                tc_max=70.,    &        !max topographic complexity
                ll_fac=1.e-6,  &        !scaling factor for lat/lon
                ssub_max=0.005          !maximum subsurface scattering signal
                                        !strength from TUW map
  
  integer(kind=2),parameter::sm_miss=-32768, &
       ssmn_max=5000  !ssm noise*100>50% (extreme noisy SSM)
  
  logical::ledge(nx,ny)
  integer,parameter::dx=1


  call date_from_days_since_year0(year0,year2,(year2-year0+1)*max_days, &
       year,mon,day)

  fpath='/mnt/scratch/stewells/h122/'
 

  call get_command_argument(1,datestamp)
  write(12,*) datestamp
  
  call system('ls /mnt/scratch/stewells/h122|grep `cat fort.12`>fort.10')
  call system('wc -l fort.10>fort.11')
  read(11,*)nf ; rewind(12) ; read(12,'(x,i4,i2.2,i2.2)')y,m,d
   print*,nf,' files on date',y,m,d

  anom = sm_miss
  call read_sm_clim(nx,ny,npass,m,d,sm_clim,sm_miss)

  ! read in TUW map of subsurface backscatter strength
  
  !open(1,file='/users/global/cmt/ASCAT/ssub_01deg.gra',form='unformatted',&
  open(1,file='/mnt/prj/swift/ASCAT_H122/ancil/ssub_01deg.gra',form='unformatted',&
     access='direct',recl=2*nx*ny)
  read(1,rec=1) ssub ; close(1)


  do f=1,nf
     read(10,'(a106)')fname
    ! print*,f,trim(fname)
     
     status = nf90_open(path=trim(fpath)//trim(fname), &
          mode=nf90_nowrite,ncid=ncid)
     if(status.ne.0) print*,nf90_strerror(status)
     

     status=nf90_inquire(ncid,ndims,nvars,nglobalatts,unlimdimid)
     if(status.ne.0) print*,nf90_strerror(status)

     if(ndims.gt.1) then
        !        print*,'only expecting 1 dimension in file' ; stop
                print*,'only expecting 1 dimension in file' ; cycle
     endif
     
     do k=1,ndims
        status=nf90_inquire_dimension(ncid=ncid,dimid=k,name=dim_name,len=len)
     enddo

     status=nf90_inq_varid(ncid,'latitude',varid)
     status=nf90_get_var(ncid,varid,ilat(1:len))
     if(status.ne.0) print*,nf90_strerror(status)
     
     status=nf90_inq_varid(ncid,'longitude',varid)
     status=nf90_get_var(ncid,varid,ilon(1:len))
     if(status.ne.0) print*,nf90_strerror(status)

     if(maxval(ilat(1:len))*ll_fac.lt.lat1 .or. &
          minval(ilat(1:len))*ll_fac.gt.lat2 .or. &
          maxval(ilon(1:len))*ll_fac.lt.lon1 .or. &
          minval(ilon(1:len))*ll_fac.gt.lon2) then
        status = nf90_close(ncid)
        cycle
     endif

     status=nf90_inq_varid(ncid,'surface_soil_moisture',varid)
     status=nf90_get_var(ncid,varid,sm_in(1:len))
     if(status.ne.0) print*,nf90_strerror(status)
     
     status=nf90_inq_varid(ncid,'time',varid)
     status=nf90_get_var(ncid,varid,time(1:len))

     status=nf90_inq_varid(ncid,'as_des_pass',varid)
     status=nf90_get_var(ncid,varid,ipass(1:len))

     status=nf90_inq_varid(ncid,'surface_soil_moisture_noise',varid)
     status=nf90_get_var(ncid,varid,ssmn(1:len))

     status=nf90_inq_varid(ncid,'surface_soil_moisture_sensitivity',varid)
     status=nf90_get_var(ncid,varid,ssms(1:len))

!     status=nf90_inq_varid(ncid,'subsurface_scattering_probability',varid)
!     status=nf90_get_var(ncid,varid,ssp(1:len))

     status=nf90_inq_varid(ncid,'topographic_complexity',varid)
     status=nf90_get_var(ncid,varid,tc(1:len))

     status=nf90_inq_varid(ncid,'surface_flag',varid)
     status=nf90_get_var(ncid,varid,sflag(1:len))
     if(status.ne.0) print*,k,'sflag ',nf90_strerror(status)

     ledge=.false.
     
!     do l=1,len
!        i = nint((ilon(l)*ll_fac-lon1)/del) + 1
!        j = nint((ilat(l)*ll_fac-lat1)/del) + 1      
!        if(i.lt.1.or.j.lt.1.or.i.gt.nx.or.j.gt.ny) cycle
!        if(sm_in(l).eq.sm_miss.and.sflag(l).eq.0) &
!             ledge(max(i-dx,1):min(i+dx,nx),j) = .true.
!     enddo

! loop over each pixel and allocate sm to relevant date, overpass and position
! on global grid

     do l=1,len
        utc_lt=ilon(l)*ll_fac/360.*24.
        i = nint((ilon(l)*ll_fac-lon1)/del) + 1
        j = nint((ilat(l)*ll_fac-lat1)/del) + 1

        if(i.lt.1.or.j.lt.1.or.i.gt.nx.or.j.gt.ny) cycle

        if(ssub(i,j)/100..gt.ssub_max) cycle
        
        if(sm_in(l).eq.sm_miss) cycle
        if(ssmn(l).ge.ssmn_max) cycle
        if(ssms(l).lt.ssms_min) cycle
!        if(ssp(l).gt.ssp_max) cycle
        if(tc(l).gt.tc_max) cycle

        if(ledge(i,j)) cycle ! exclude data within dx pixels of swath edge
        
        yy = year(floor(time(l)))
        mm = mon(floor(time(l)))
        dd = day(floor(time(l)))
        if(yy.ne.y.or.mm.ne.m.or.dd.ne.d) cycle


        lt = 24*(time(l)-floor(time(l))) + utc_lt
        if(lt.lt.0. .or. lt.ge.24.) call solar_date(lt,yy,mm,dd)
!
! remove pixels where local overpass time more than dt_min from 9:30
! dt_min was 2 hours in H119 code now reduced to 1.5 hours - this removes
! late stage metopA data 
!
        if(ipass(l).eq.0) then !ascending pm overpass
           if ( abs(lt - (lt_eq+12.)) .ge. dt_min) cycle
        elseif(ipass(l).eq.1) then  !descending am overpass
           if ( abs(lt -  lt_eq     ) .ge. dt_min) cycle
        endif

        smtot(i,j,ipass(l)+1) = smtot(i,j,ipass(l)+1) + real(sm_in(l))
        ntot(i,j,ipass(l)+1)  = ntot(i,j,ipass(l)+1) + 1.
        
     enddo
     
     

  enddo

  where(ntot.gt.0)
     daily(:,:,:,2) = nint(smtot/ntot)
     anom = daily(:,:,:,2) - nint(sm_clim)
  elsewhere
     daily(:,:,:,2) = sm_miss
     anom = sm_miss
  endwhere
  where(sm_clim.eq.sm_miss) anom = sm_miss
  
  daily(:,:,:,1) = nint(ntot)
  
  ystr = datestamp(1:4)
  mstr = datestamp(5:6)
  dstr = datestamp(7:8)

  open(1,file='/mnt/prj/swift/ASCAT_H122/&
       H122_vn2J_daily/'//ystr//'/'//mstr//'/ASCAT_sm_'//datestamp//'.gra',&
       form='unformatted',access='direct',recl=2*nx*ny*2*npass)
  write(1,rec=1) daily ; close(1)
  !
  open(1,file='/mnt/prj/swift/ASCAT_H122/&
       H122_vn2J_daily/'//ystr//'/'//mstr//'/ASCAT_dsm_'//datestamp//'_am.gra',&
       form='unformatted',access='direct',recl=2*nx*ny,iostat=iop)

  write(1,rec=1,iostat=iop) anom(:,:,2)

  write(1,rec=1) anom(:,:,2) ; close(1)
  open(1,file='/mnt/prj/swift/ASCAT_H122/&
       H122_vn2J_daily/'//ystr//'/'//mstr//'/ASCAT_dsm_'//datestamp//'_pm.gra',&
       form='unformatted',access='direct',recl=2*nx*ny)
  write(1,rec=1) anom(:,:,1) ; close(1)

  call system('rm fort.10 fort.11 fort.12')
  
end program grid_NRT_vn2J

!
!---------------------------------------------------------------------
!
subroutine date_from_days_since_year0(year0,year2,ndays,year,mon,day)
implicit none
!
! computes date for full number of days since 0Z 1 Jan in year0
! (0 days is 1/1/1900, 1 day is 2/1/1900...)
integer::year0,year2,ndays,ileap,yy,mm,dd,k,k0
integer,dimension(ndays)::year,mon,day
integer,parameter::nmon=12
integer::first_day_of_mon(nmon+1)
data first_day_of_mon/1,32,60,91,121,152,182,213,244,274,305,335,366/
!
yy=year0 ; mm=1 ; dd=0 ; k0=0
!
do k=0,ndays
!
  dd=dd+1
!
  ileap=0
  if(mod(yy,4).eq.0 .and. yy.ne.1900 .and. mm.ge.2) ileap=1
!
  if(k-k0+1.ge.first_day_of_mon(mm+1)+ileap) then
    mm=mm+1 ; dd=1
    if(mm.gt.nmon) then
      yy=yy+1 ; mm=1 ; k0=k
    endif
  endif
  if(k.gt.0) then
    year(k) = yy
    mon(k)  = mm
    day(k)  = dd
  endif
!
enddo
!
return
end
!
!---------------------------------------------------------------------
!
subroutine solar_date(lt,yy,mm,dd)
implicit none
!
! adjusts local time and date when local solar date differs from UTC date
!
real::lt
integer,parameter::nmon=12
integer::yy,mm,dd,ileap,ndays(nmon)
data ndays/31,28,31,30,31,30,31,31,30,31,30,31/
!
if(lt.lt.0.) then          ! this can happen for evening overpasses in western
                           ! hemisphere
  lt = lt+24.
  dd = dd-1
!
  if(dd.lt.1) then
    if(mm.gt.1) then
      dd=ndays(mm-1)
      mm=mm-1
      if(mm.eq.2.and.mod(yy,4).eq.0) dd=dd+1
    else
      yy=yy-1
      dd=ndays(nmon)
      mm=nmon
    endif
  endif
!
elseif(lt.ge.24.) then

  lt = lt-24.
  dd = dd+1
!
  ileap= 0 ; if(mod(yy,4).eq.0 .and. mm.eq.2) ileap=1
  if(dd.gt.ndays(mm)+ileap) then
    dd=1
    if(mm.lt.nmon) then
      mm=mm+1
    else
      mm=1
      yy=yy+1
    endif
  endif

endif
!
return
end
!
!---------------------------------------------------------------------
!
subroutine read_sm_clim(nx,ny,npass,m,d,sm_clim,sm_miss)
  implicit none
  !
  ! for given month and day of month reads pre-existing climatology on
  ! nx x ny grid
  ! code assumes 31 days in month when interpolating between adjacent months
  ! originally (Dec 24) this code set climatology to undefined if either month
  ! was missing. This threw out a lot of data so now assume the month where there
  ! is non-missing data provides the climatology
  !
  integer::nx,ny,npass,m,d
  real::sm_clim(nx,ny,npass),wt
  integer(kind=2),dimension(nx,ny)::clim,climm(nx,ny)
  integer::m1,p,mm
  integer,parameter::ndays=31
  character*100::fname
  integer(kind=2)::sm_miss


  mm = m + 1
  wt = 1. - (d-ndays/2.)/ndays
  if(d.le.real(ndays/2.)) then
     mm = m - 1  !in first half of month interpolate with previous month
     wt = 1. - (ndays/2.-d)/ndays
  endif
  
  if(mm.lt.1) mm=12
  if(mm.gt.12) mm=1

  do p=1,npass

     if(p.eq.1) then
!        fname='/prj/swift/ASCAT_H129/SSA_grid_01_vn0/clim/clim_2007_2023_pm'
!        fname='/prj/swift/ASCAT_H129/SSA_grid_01_vn2c/clim/clim_2007_2024_pm'
!        fname='/prj/swift/ASCAT_H129/SSA_grid_01_vn2H/clim/clim_2007_2024_pm'
        fname='/mnt/prj/swift/ASCAT_H122/ancil/SSA_grid_01_vn2H/clim/clim_2007_2024_pm'
     else
!        fname='/prj/swift/ASCAT_H129/SSA_grid_01_vn0/clim/clim_2007_2023_am'
!        fname='/prj/swift/ASCAT_H129/SSA_grid_01_vn2c/clim/clim_2007_2024_am'
!        fname='/prj/swift/ASCAT_H129/SSA_grid_01_vn2H/clim/clim_2007_2024_am'
        fname='/mnt/prj/swift/ASCAT_H122/ancil/SSA_grid_01_vn2H/clim/clim_2007_2024_am'
     endif
     
     print*,trim(fname)
     open(1,file=trim(fname)//'.gra',status='old',form='unformatted', &
          access='direct',recl=2*nx*ny)
          
     read(1,rec=m) clim ; read(1,rec=mm) climm
     close(1)

     where(clim.lt.0)  clim  = climm !if either month has missing values
     where(climm.lt.0) climm = clim  !use the other month

     sm_clim(:,:,p) = wt*clim + (1.-wt)*climm
     
  enddo

  return
end subroutine read_sm_clim
