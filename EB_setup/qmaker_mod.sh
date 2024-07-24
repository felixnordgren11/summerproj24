#!/bin/bash
# Creates q-file for UppASD bcc (110) surface
qlen=$1
xx=`grep ncell inpsd.dat | awk '{ print $2}'`
echo $qlen $xx
rm -f tmpfile0 tmpfile 
# neg x 
for q in `seq 0 $qlen`
do 
   echo $q | awk -v xx=$xx '{ printf "%f %f %f\n", 0.5*$1/xx,0 ,0  }' >> tmpfile 
done
# number of lines
nq=`wc tmpfile | awk '{ print $1 }'` 
echo $nq | awk '{ printf "            %i\n",$1 }' > tmpfile0 
cat tmpfile0 tmpfile > qfile
rm -f tmpfile0 tmpfile 

