#!/bin/bash
#
# Usage:  build_mac.sh [options] filename.mac
# Options: -n nevents            number of events
#          -s seed               WCSim random seed, default is set from SLURM_ARRAY_TASK_ID)
#          -g geom               WCSim geometry default is nuPRISM_mPMT)
#          -G Gd_doping          Gadolinium doping percent (default is 0)
#          -r dark-rate          dark rate [kHz] (default is 0.1 kHz)
#          -D DAQ_mac_file       WCSim daq mac file (default is [data_dir]/[name]/WCSim/macros/daq.mac
#          -N NUANCE_input_file  input text file for NUANCE format event vectors
#          -E fixed_or_max_Evis  fixed or maximum visible energy [MeV]
#         [-e min_Evis]          minimum visible energy [MeV]
#          -P particle_type      particle type
#          -o orientation        orientation of axis of tank (default is "y")
#          -p pos_type           position type [fix|unif]
#         [-x x_pos]             fixed x position (for pos_type=fix) [cm]
#          -y y_pos              fixed y position (for pos_type=fix) or maximum half-y (for pos_type=unif) [cm]
#         [-z z_pos]             fixed z position (for pos_type=fix) [cm]
#         [-R R_pos]             fixed R position (for pos_type=fix) or max R (for pos_type=unif) [cm]
#          -d dir_type           direction type [fix|2pi|4pi]
#          -u x_dir              x direction (for dir_type=fix)
#          -v y_dir              y direction (for dir_type=fix)
#          -w z_dir              z direction (for dir_type=fix)
#          -m                    turn off muon decay
#          -f output_file.root   Output root file

#for arg in "$@"; do
#  echo $arg
#done
geom="nuPRISM_mPMT"
daqfile="${WCSIMDIR}/macros/daq.mac"
seed=${SLURM_ARRAY_TASK_ID}
orient="y"
while getopts "n:s:g:G:r:D:N:E:e:P:o:p:x:y:z:R:d:u:v:w:i:f:mFL" flag; do
  case $flag in
    n) nevents="${OPTARG}";;
    s) seed="${OPTARG}";;
    g) geom="${OPTARG}";;
    G) gad="${OPTARG}";;
    r) darkrate="${OPTARG}";;
    D) daqfile="$(readlink -f "${OPTARG}")";;
    N) nuance="$(readlink -f "${OPTARG}")";;
    E) Emax="${OPTARG}";;
    e) Emin="${OPTARG}";;
    P) pid="${OPTARG}";;
    o) orient="${OPTARG}";;
    p) pos="${OPTARG}";;
    x) xpos="${OPTARG}";;
    y) ypos="${OPTARG}";;
    z) zpos="${OPTARG}";;
    R) rpos="${OPTARG}";;
    d) dir="${OPTARG}";;
    u) xdir="${OPTARG}";;
    v) ydir="${OPTARG}";;
    w) zdir="${OPTARG}";;
    i) procs+=("${OPTARG}");;
    f) rootfile="$(readlink -m "${OPTARG}")";;
    m) nomichel=1
  esac
done
shift $((OPTIND - 1))
file="$(readlink -m "$1")"
if [ -z $nevents ]; then echo "Number of events not set"; exit 1; fi
if [ -z $seed ]; then echo "Random seed not set"; exit 1; fi
if [ -z $geom ]; then echo "Geometry not set"; exit 1; fi
if [ -z $daqfile ]; then echo "DAQ mac file not set"; exit 1; fi
if [ -z $rootfile ]; then echo "Root file not set"; exit 1; fi
if [ -z $file ]; then echo "Output mac file name not set"; exit 1; fi
if [ ! -z $nuance ]; then
  if [ ! -z $Emax ]; then echo "Using nuance file but Emax is set"; exit 1; fi
  if [ ! -z $Emin ]; then echo "Using nuance file but Emin is set"; exit 1; fi
  if [ ! -z $pid  ]; then echo "Using nuance file but PID is set"; exit 1; fi
  if [ ! -z $pos ]; then echo "Using nuance file but pos is set"; exit 1; fi
  if [ ! -z $xpos ]; then echo "Using nuance file but x pos is set"; exit 1; fi
  if [ ! -z $ypos ]; then echo "Using nuance file but y pos is set"; exit 1; fi
  if [ ! -z $zpos ]; then echo "Using nuance file but z pos is set"; exit 1; fi
  if [ ! -z $rpos ]; then echo "Using nuance file but r pos is set"; exit 1; fi
  if [ ! -z $dir  ]; then echo "Using nuance file but dir type is set"; exit 1; fi
  if [ ! -z $xdir ]; then echo "Using nuance file but x dir is set"; exit 1; fi
  if [ ! -z $ydir ]; then echo "Using nuance file but y dir is set"; exit 1; fi
  if [ ! -z $zdir ]; then echo "Using nuance file but z dir is set"; exit 1; fi
else
  if [ -z $Emax ]; then echo "Energy not set"; exit 1; fi
  if [ -z $pid  ]; then echo "PID not set"; exit 1; fi
  if [ -z $dir  ]; then echo "Direction type not set"; exit 1; fi
  if [ "$dir" == "fix" ]; then
    if [ -z $xdir ]; then echo "Dir is fix but x dir not set"; exit 1; fi
    if [ -z $ydir ]; then echo "Dir is fix but y dir not set"; exit 1; fi
    if [ -z $zdir ]; then echo "Dir is fix but z dir not set"; exit 1; fi
  else
    if [[ "$dir" != [24]pi ]]; then echo "Unrecognised direction type"; exit; fi
    if [ ! -z $xdir ]; then echo "Dir is $dir but x dir set"; exit; fi
    if [ ! -z $ydir ]; then echo "Dir is $dir but y dir set"; exit; fi
    if [ ! -z $zdir ]; then echo "Dir is $dir but z dir set"; exit; fi
  fi
  if [ "$pos" == "unif" ]; then
    if [ ! -z $xpos ]; then echo "Pos is unif but x pos set"; exit 1; fi
    if [ ! -z $zpos ]; then echo "Pos is unif but z pos set"; exit 1; fi
    if [ -z $ypos ]; then echo "Pos is unif but max half-y not set"; exit; fi
    if [ -z $rpos ]; then echo "Pos is unif but max R not set"; exit; fi
  elif [ "$pos" == "fix" ]; then
    if [ ! -z $rpos ]; then
      if [ ! -z $xpos ]; then echo "Both R pos and x pos set"; exit; fi
      if [ ! -z $zpos ]; then echo "Both R pos and z pos set"; exit; fi
      xpos=$rpos
      zpos=0
    else
      if [ -z $xpos ]; then echo "Neither R pos nor x pos set"; exit 1; fi
      if [ -z $zpos ]; then echo "Neither R pos not z pos set"; exit 1; fi
    fi
    if [ -z $ypos ]; then echo "y pos not set"; exit 1; fi
  fi
fi

case $orient in
  y) rot1="1 0 0"
     rot2="0 0 1"
     ;;
  x) rot1="0 1 0"
     rot2="0 0 1"
     ;;
  z) rot1="1 0 0"
     rot2="0 1 0"
     ;;
  *) echo "Unknown orientation"
     exit 1
     ;;
esac

if [ -z "${nuance}" ]; then
  # Calculate Ekin from Evis
  declare -A EKth
  EKth[e-]=0.264
  EKth[e+]=0.264
  EKth[mu-]=54.6
  EKth[mu+]=54.6
  m_e=0.511
  EKth[gamma]="$(python -c "print(2*(${m_e}+${EKth[e-]}))")"
  EKth[pi-]=72.1
  EKth[pi+]=72.1
  EKth[pi0]="$(python -c "print(2*${EKth[gamma]})")"
  EkinMax="$(python -c "print(${Emax}+${EKth[${pid}]:-0})")"
  [ ! -z "${Emin}" ] && EkinMin="$(python -c "print(${Emin}+${EKth[${pid}]:-0})")"
  if [ "${pid}" == "gamma" ]; then
    generator="gamma-conversion"
  else
    generator="gps"
  fi
else
  generator="muline"
fi

echo     "/run/verbose                           1"                    > "${file}"
echo     "/tracking/verbose                      0"                    >> "${file}"
echo     "/hits/verbose                          0"                    >> "${file}"
echo     "/random/setSeeds                       $seed $seed"          >> "${file}"
echo     "/WCSim/WCgeom                          $geom"                >> "${file}"
if [ ! -z "${gad}" ]; then
  if [ ! "$gad" == 0 ]; then
    echo "/WCSim/DopedWater                      true"                 >> "${file}"
    echo "/WCSim/DopingConcentration             $gad"                 >> "${file}"
  else
    echo "/WCSim/DopedWater                      false"                >> "${file}"
  fi
fi
echo     "/WCSim/Construct"                                            >> "${file}"
echo     "/WCSim/PMTQEMethod                     Stacking_Only"        >> "${file}"
echo     "/WCSim/PMTCollEff                      on"                   >> "${file}"
echo     "/WCSim/SavePi0                         false"                >> "${file}"
echo     "/DAQ/Digitizer                         SKI"                  >> "${file}"
echo     "/DAQ/Trigger                           NDigits"              >> "${file}"
echo     "/control/execute                       $daqfile"             >> "${file}"
if [ ! -z $darkrate ]; then
  echo   "/DarkRate/SetDarkRate                  $darkrate kHz"        >> "${file}"
  echo   "/DarkRate/SetDarkMode                  1"                    >> "${file}"
  echo   "/DarkRate/SetDarkHigh                  100000"               >> "${file}"
  echo   "/DarkRate/SetDarkLow                   0"                    >> "${file}"
  echo   "/DarkRate/SetDarkWindow                4000"                 >> "${file}"
fi
for proc in "${procs[@]}" ; do
  echo   "/process/inactivate                    ${proc}"              >> "${file}"
done
if [ "${nomichel}" == 1 ]; then
  echo   "/particle/select                       mu-"                  >> "${file}"
  echo   "/particle/process/inactivate           7"                    >> "${file}"
  echo   "/particle/process/inactivate           8"                    >> "${file}"
fi
echo     "/mygen/generator                       $generator"           >> "${file}"
if [ ! -z $nuance ]; then
  echo   "/mygen/vecfile                         $nuance"              >> "${file}"
else
  echo   "/gps/particle                          $pid"                 >> "${file}"
  if [ ! -z $Emin ]; then
    echo "/gps/ene/type                          Lin"                  >> "${file}"
    echo "/gps/ene/intercept                     1"                    >> "${file}"
    echo "/gps/ene/min                           $EkinMin MeV"         >> "${file}"
    echo "/gps/ene/max                           $EkinMax MeV"         >> "${file}"
  else
    echo "/gps/energy                            $EkinMax MeV"         >> "${file}"
  fi
  if [ "$dir" == "fix" ]; then
    echo "/gps/direction                         $xdir $ydir $zdir"    >> "${file}"
  elif [ "$dir" == "2pi" ]; then
    echo "/gps/ang/type                          iso"                  >> "${file}"
    echo "/gps/ang/rot1                          $rot1"                >> "${file}"
    echo "/gps/ang/rot2                          $rot2"                >> "${file}"
    echo "/gps/ang/mintheta                      90  deg"              >> "${file}"
    echo "/gps/ang/maxtheta                      90  deg"              >> "${file}"
    echo "/gps/ang/minphi                        0   deg"              >> "${file}"
    echo "/gps/ang/maxphi                        360 deg"              >> "${file}"
  elif [ "$dir" == "4pi" ]; then
    echo "/gps/ang/type                          iso"                  >> "${file}"
  fi
  if [ "$pos" == "unif" ]; then
    echo "/gps/pos/type                          Volume"               >> "${file}"
    echo "/gps/pos/shape                         Cylinder"             >> "${file}"
    echo "/gps/pos/rot1                          $rot1"                >> "${file}"
    echo "/gps/pos/rot2                          $rot2"                >> "${file}"
    echo "/gps/pos/radius                        $rpos cm"             >> "${file}"
    echo "/gps/pos/halfz                         $ypos cm"             >> "${file}"
  else
    echo "/gps/position                          $xpos $ypos $zpos cm" >> "${file}"
  fi
fi
echo     "/Tracking/fractionOpticalPhotonsToDraw 0.0"                  >> "${file}"
echo     "/WCSimIO/RootFile                      $rootfile"            >> "${file}"
echo     "/WCSimIO/SaveRooTracker                0"                    >> "${file}"
echo     "/run/beamOn                            $nevents"             >> "${file}"

