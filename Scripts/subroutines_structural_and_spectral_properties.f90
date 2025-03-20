
!@@@@@@@@@@ Subroutine for the generation of random numbers@@@@@@@@@@
subroutine init_random_seed()
implicit none
integer:: i,n, clock
integer, dimension(:),allocatable:: seed

call random_seed(size=n)
allocate(seed(n))
call system_clock(count=clock)
seed = clock + 37* (/ ( i - 1, i = 1, n)  /)
call random_seed(put = seed)
deallocate (seed)
end


!@@@@@@@@Subroutine for average degree calculation@@@@@@@@@@@@@@@@@
subroutine AverageDegree(N,M,Degreeprom) 
! - N is the size of the network (or number of nodes) (input)
! - M is the binary adjacency matrix (input)
! Degreeporm is the average degree output

IMPLICIT NONE
INTEGER(KIND=2), PARAMETER:: double = 8
INTEGER, INTENT(IN) :: N
!REAL(double), DIMENSION(N), INTENT(IN) :: dd
REAL(double), DIMENSION(N,N), INTENT(IN) :: M
REAL(double), INTENT(OUT) :: Degreeprom
REAL(double), PARAMETER :: ZERODD = 0_double
REAL(double) :: Degree, DegreeTyp
INTEGER :: i

Degreeprom = ZERODD
!DegreepromTyp = ZERODD
do i = 1, N
	Degree = sum(M(i,:))
	Degreeprom = Degreeprom + Degree
end do

Degreeprom = Degreeprom/N

end subroutine AverageDegree


!@@@@@@@@@@@@@Subroutine for number of nonisolated vertices calculatio@@@@
subroutine Vertexes(dd,N,B,Verprom)
! dd - alloccates the degrees of the network (degree of every node in an 1D array)
! N - the number of nodes
! B - the binary adjacency matrix
! Verprom - the number of nonisolated vertices (output)

IMPLICIT NONE
INTEGER(KIND=2), PARAMETER:: double = 8
INTEGER, INTENT(IN) :: N
REAL(double), DIMENSION(N,N), INTENT(IN) :: B
REAL(double), DIMENSION(N), INTENT(IN) :: dd
REAL(double), INTENT(OUT) :: Verprom
REAL(double) :: VER

REAL(double), PARAMETER :: ZERODD = 0_double
REAL(double), PARAMETER :: ONEDD = 1_double
INTEGER :: i

VER = ZERODD

do i = 1, N
	 ! Added in case of having a directed network
	if ((dd(i) .ne. ZERODD))then!.OR. (ddd(i).ne.ZERODD) ) then 
	! Added condition .OR. if we have a directed network
		VER = VER + ONEDD
	end if
end do
Verprom = VER
		
end subroutine Vertexes



!@@@@@@Subroutine ratio between between near- @@@@@@@@@@@@@@@@@@
!and next-to nearest neighbor eigenvalue
!(complex or real)@@@@@@@
subroutine RatioCom(N, W, RTprom)
! N - the number of nodes
! W - 1D array containing the ordered eigenvalues of the network (if spectrum is real, 
!if not eigenvalues are not ordered) 
! RTprom - average ration between near- and next-to nearest neighbor eigenvalue} (output)
IMPLICIT NONE
INTEGER(KIND=2), PARAMETER:: double = 8
INTEGER, INTENT(IN) :: N
!COMPLEX(double), DIMENSION(N), INTENT(IN) :: W  !!Commented LINE
REAL(double), DIMENSION(N), INTENT(IN) :: W
REAL(double), Dimension(N):: DD
REAL(double), INTENT(OUT) :: RTprom
REAL(double) :: RT, minn, maxx, a, b
REAL(double), PARAMETER :: ZERODD = 0_double
REAL(double), PARAMETER :: ONEDD = 1_double
REAL(double), PARAMETER :: TWODD = 2_double
INTEGER :: I, J

RT = ZERODD

!If you're treating with real spectra (out of nonHermitian matrix), comment the stuff
!below and uncomment the line "Commented LINE" above

DO i = 1, N
	do j = 1, N
		if (i.ne.j) then
		DD(j) = abs(W(i)-W(j))
		else
		DD(j) = 10e6
		end if
	end do
	minn = Minval(DD)
	maxx = MINval(DD,mask=DD.gt.minn)
	
	RT = RT + minn/maxx

END DO
	RTprom = RT
		
end subroutine RatioCom



!@@@@@Subroutine Shannon entropy and participation ratio calculation@@@@@@@@@
subroutine Shannon(N, A, Shannonprom, IPRprom)
! N - number of nodes
! A - matrix containing the eigenvectors of the network per column as A(:,i)
! Shannonprom - average Shannon entropy (output)
! IPRprom - average participation ratio (output)
IMPLICIT NONE
INTEGER(KIND=2), PARAMETER:: double = 8
INTEGER, INTENT(IN) :: N
COMPLEX(double), DIMENSION(N,N), INTENT(IN) :: A
REAL(double), INTENT(OUT) :: Shannonprom, IPRprom
REAL(double) :: S, IP
REAL(double), PARAMETER :: ZERODD = 0_double
REAL(double), PARAMETER :: ONEDD = 1_double
REAL(double), PARAMETER :: TWODD = 2_double
INTEGER :: i

S = ZERODD
IP = ZERODD
Do i = 1, N
			S = S - dot_product(abs(A(:,i))**2, dlog(abs(A(:,i))**2))
			IP = IP + ONEDD/(sum(abs(A(:,i))**4))
End Do
Shannonprom = S
IPRprom = IP
end subroutine Shannon


!@@@@@@Subroutine ratio between consecutive level spacings (real spectra)@@@@@@
subroutine Ratio(N, W, RTprom)
! N- number of nodes
! W - 1D array containing the eigenvalues of the network
! RTprom - average ration between consecutive eigenvalue level spacings
IMPLICIT NONE
INTEGER(KIND=2), PARAMETER:: double = 8
INTEGER, INTENT(IN) :: N
REAL(double), DIMENSION(N), INTENT(IN) :: W
REAL(double), INTENT(OUT) :: RTprom
REAL(double) :: RT, minn, maxx, a, b
REAL(double), PARAMETER :: ZERODD = 0_double
REAL(double), PARAMETER :: ONEDD = 1_double
REAL(double), PARAMETER :: TWODD = 2_double
INTEGER :: I

RT = ZERODD
do I = 2, N-1
	a = W(I+1)-W(I)
	b = W(I)-W(I-1)
	if (a.lt.b)then
	minn = a
	maxx = b
	else
	minn = b
	maxx = a
	end if
	RT = RT + minn/maxx
end do
RTprom = RT
		
end subroutine Ratio



!@@@@@@@Subroutine topological indices (Randic and Harmonic indices)@@@@@@@@
subroutine Randic(dd,N,B,Randicprom, Harmonicprom)
! dd - 1D array containing the degrees of every node in the network
! N- number of nodes
! B - binary adjacency matrix
! Randicprom - average Randic index
! Harmonicprom - average Harmonic index
IMPLICIT NONE
INTEGER(KIND=2), PARAMETER:: double = 8
INTEGER, INTENT(IN) :: N
REAL(double), DIMENSION(N), INTENT(IN) :: dd
REAL(double), DIMENSION(N,N), INTENT(IN) :: B
REAL(double), INTENT(OUT) :: Randicprom, Harmonicprom
REAL(double) :: RR, RC, HH, H

!REAL(double), DIMENSION(N):: dd
REAL(double), PARAMETER :: ZERODD = 0_double
REAL(double), PARAMETER :: ONEDD = 1_double
REAL(double), PARAMETER :: TWODD = 2_double
INTEGER :: i, j

RR = ZERODD
HH = ZERODD
	
do i = 1, N-1
	RC = ZERODD
	H = ZERODD
	do j = i+1, N
		if (B(i,j).ne. 0) then
			RC = RC + ONEDD / ( sqrt(dd(i)*dd(j)) )
			H = H + TWODD/( ( dd(i) + dd(j) ) )
		end if
	end do
	RR = RR + RC
	HH = HH + H
end do
Randicprom = RR
Harmonicprom = HH
		
end subroutine Randic
