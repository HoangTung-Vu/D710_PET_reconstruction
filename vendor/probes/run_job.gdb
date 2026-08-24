# Inject a cmpclient .job into pet_recon and run GE's own correction pipeline.
#
# Prerequisites, in order:
#   job2gdb.py <file.job> <writable-overlap-dir>  > job.gdb
#   run_petsw.sh run_job.gdb
#
# Set OUTDIR below to where the debug dumps should land (gdb's "set cwd").
source /vendor/boot.gdb

# Debug masks, read out of sharcCmpShowDebOpts():
#   0x001 general      0x002 AP buffer setup   0x004 LOAD      0x008 RDF
#   0x010 JOB          0x020 REPLY             0x040 ROI       0x080 AP_CONFIG
#   0x100 AP_RANDOMS   0x200 AP_FORE           0x400 AP_FPEXCEPTION
# 0x100 is the one that makes the randoms sinogram reach disk.
set var sharcCmpDebugFlag = 0x1FF

# CPCCommThread was killed at boot, so fdCpcIgFIFO -- the CPC->IG FIFO that
# sharcCmpProcessJobOnAp writes an 8-byte "job ready" message to at
# cpcApLib.cpp:412 -- is still -1, and the write() failure makes the whole
# function bail to cpcApLib.cpp:788 returning -1 before doing any work.
# A plain pipe satisfies it.
set $p = (int *) malloc(8)
print (int) pipe($p)
set var fdCpcIgFIFO = $p[1]

source /vendor/job.gdb

# Opens every RDF named in the job, reads their headers, and fills the apCfg
# and sysGeometry globals.  Returns 0 on success.  gdb needs the explicit cast
# because S_HOST_CMP_JOB_REQ is an anonymous struct in the DWARF.
print ((int (*)(void *)) sharcCmpOpenDataFiles)(&IgJobReq)

print apCfg.normDeadtimeSlopeInnerRing
print apCfg.normDeadtimeSlopeOuterRing
print apCfg.scatScaleFactorLimit
print apCfg.norm3dGeometryPeriod
print apCfg.norm3dCorrXtalEffClipValue
print apCfg.scatCountRateFuncCoeff

set $b = (char *) malloc(64)
echo \n=== sharcCmpProcessJobOnAp packet 0 ===\n
print ((int (*)(void *, unsigned int, char *)) sharcCmpProcessJobOnAp)(&IgJobReq, 0, $b)
