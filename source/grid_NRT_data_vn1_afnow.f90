program grid_NRT_data
use netcdf
implicit none
!
! copied from pre-operational test_data.f90
! puts 10 minute LI flash data fron netcdf files on
! regular grid every 15 minutes
!
! USES TIMESTAMP TO CORRECTLY ALLOCATE ALL FLASHES TO 15 MINUTE TIME STEPS
! THIS VERSION READS THREE MOST RECENT NETCDF FILES
!
! gfortran grid_NRT_data_vn1_afnow.f90 $(nf-config --fflags) $(nf-config --flibs) -o grid_NRT_LI_afnow
integer::status,ncid,varid,ndims,nvars,nglobalatts,unlimdimid,len,numdims
character*148::fn1,fn2,dim_name,fname
!character*28::path='download_dir'
!character*28::path='/scratch/cmt/MTG_NRT'
character*40::path='/mnt/scratch/stewells/MTG_LI0691/tidy'
!character*147::path,fn1,fn2,dim_name,fname
character*12::datestamp(2)
integer::i,j,k,nflash,iop,f,year,mon,day,hr,min,step,last_step,step2,year2,mon2,day2,hr2,min2,fstep,l
real,parameter::sf=0.0027,undef=-999.9, &
                del=0.05,lon1=-20.+del/2.,lat1=-36.+del/2.
integer,parameter::big=100000,npd=96,nx=1400,ny=1440,secs_per_day=3600*24,nfiles=3
integer(kind=2),dimension(big)::flat,flon
!
!real,dimension(nx,ny,npd)::gflash
!integer(kind=2),dimension(nx,ny,2)::existing_data,gflash
integer(kind=2),dimension(nx,ny,npd)::gflash
real(kind=8),dimension(big)::ftime
character*4::chyear
character*2::chmon
!
last_step=0
gflash=0.

! identify 3 most recent filenames

!!!call system('ls -lt /mnt/scratch/cmt/MTG_NRT/*BODY*.nc|cut -d"/" -f5|head -3>b.tmp')
call system('ls -lt /mnt/scratch/stewells/MTG_LI0691/tidy/*BODY*.nc|cut -d"/" -f7|head -3>b.tmp')

open(10,file='b.tmp')
!
! read latest filename to identify time steps to update
!
read(10,'(101x,i4,i2.2,i2.2,i2.2,i2.2,3x,i4,i2.2,i2.2,i2.2,i2.2)',iostat=iop) year,mon,day,hr,min,&
     year2,mon2,day2,hr2,min2
!
if(iop.ne.0)then
  print*,'problem with length of filename!!!'
  print*,fname ; stop
endif
!
step = hr*4 + ceiling((min+1)/15.)
step2 = hr2*4 + ceiling((min2+1)/15.)
write(datestamp(1),'(i4,i2.2,i2.2,i2.2,i2.2)') year,mon,day,hr,mod(step-1,4)*15
write(datestamp(2),'(i4,i2.2,i2.2,i2.2,i2.2)') year2,mon2,day2,hr2,mod(step2-1,4)*15
print*, '**'
print*,'start/end time in latest filename'
print*,year,mon,day,hr,min,hr2,min2
print*,'time step(s) to update'
print*,datestamp(1),' ',datestamp(2),' ',step,step2
print*, "**"
write(chyear,'(i4)') year ; write(chmon,'(i2.2)') mon
!
rewind(10)
!
do f=1,nfiles
   !
   read(10,'(a148)',iostat=iop) fname
   !
   print*,f,trim(path)//'/'//trim(fname) 

   status = nf90_open(path=trim(path)//'/'//trim(fname),mode=nf90_nowrite, &
        ncid=ncid)
  if(status.ne.0) print*,nf90_strerror(status)
  if(status.ne.0) print*,trim(path)//'/'//trim(fname)
  !
  status=nf90_inquire(ncid,ndims,nvars,nglobalatts,unlimdimid)
  if(status.ne.0) print*,nf90_strerror(status)
!
  do k=1,ndims
    status=nf90_inquire_dimension(ncid=ncid,dimid=k,name=dim_name,len=len)
    if(trim(dim_name).eq.'flashes') nflash=len
  enddo
!
  status=nf90_inq_varid(ncid,'latitude',varid)
  status=nf90_get_var(ncid,varid,flat(1:nflash))
  if(status.ne.0) print*,nf90_strerror(status)
!
  status=nf90_inq_varid(ncid,'longitude',varid)
  status=nf90_get_var(ncid,varid,flon(1:nflash))
  if(status.ne.0) print*,nf90_strerror(status)
!
  status=nf90_inq_varid(ncid,'flash_time',varid)
  status=nf90_get_var(ncid,varid,ftime(1:nflash))
  if(status.ne.0) print*,nf90_strerror(status)
  !
  print*,'file number ',f,fname(110:130)
  print*,'number of flashes in file and min/max seconds',nflash, &
       nint(minval(ftime(1:nflash))),nint(maxval(ftime(1:nflash)))
  do k=1,nflash
    i=nint((flon(k)*sf-lon1)/del)+1
    j=nint((flat(k)*sf-lat1)/del)+1
    if(i.lt.1.or.i.gt.nx.or.j.lt.1.or.j.gt.ny) cycle
!
    fstep = floor(mod(floor(ftime(k)),secs_per_day)/(real(secs_per_day)/npd)) + 1
!    if(k.lt.10)print*,k,flon(k),flat(k),i,j,fstep,step,step2,ftime(k)/secs_per_day
    
!
    if(fstep.eq.step.or.fstep.eq.step2) then
      if(mod(k,50000).eq.1)print*,k,mod(floor(ftime(k)),3600),fstep,step,step2
      gflash(i,j,fstep) = gflash(i,j,fstep) + 1
    endif

  enddo

  status = nf90_close(ncid)
  if(status.ne.0) print*,nf90_strerror(status)
  print*,'file number',f,' max flashes in step/step2',maxval(gflash(:,:,step)),maxval(gflash(:,:,step2))
  print*, '**'
!
enddo
!
open(1,file='/mnt/scratch/stewells/MTG_LI0691/flash_count_NRT/'//datestamp(1)//'.gra',&
       form='unformatted',access='direct',recl=2*nx*ny)
write(1,rec=1) gflash(:,:,step) ; close(1)
!
open(1,file='/mnt/prj/swift/MTG_LI_flash_count/'//chyear//'/'//chmon//'/'//datestamp(1)//'.gra',&
       form='unformatted',access='direct',recl=2*nx*ny)
write(1,rec=1) gflash(:,:,step) ; close(1)
!
if(datestamp(2).ne.datestamp(1)) then
  open(1,file='/mnt/scratch/stewells/MTG_LI0691/flash_count_NRT/'//datestamp(2)//'.gra',&
         form='unformatted',access='direct',recl=2*nx*ny)
  write(1,rec=1) gflash(:,:,step2) ; close(1)
endif
!
call system('rm b.tmp')
!
end
