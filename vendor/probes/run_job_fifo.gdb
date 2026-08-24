set cwd /tmp/claude-1000/-home-hoangtungvm-UET-Handson-PET-CT-Reconstruction/ea2cccb1-3402-49e4-948d-8c25722a4447/scratchpad/out
source /vendor/boot.gdb
set var sharcCmpDebugFlag = 0x100
# Open BOTH halves of the CPC<->IG FIFO pair O_RDWR (=2) from inside the
# inferior.  O_RDWR never blocks and never returns ENXIO, so neither side's
# own open() can stall, and fdCpcIgFIFO becomes a genuinely writable fd whose
# reader is the IG main thread -- not a dead-end pipe.
set $figc = (int) open("IG_CPC_FIFO", 2)
set $fcgi = (int) open("CPC_IG_FIFO", 2)
print $figc
print $fcgi
set var fdCpcIgFIFO = $fcgi
source /vendor/job.gdb
print ((int (*)(void *)) sharcCmpOpenDataFiles)(&IgJobReq)
set $b = (char *) malloc(64)
# Call from a WORKER thread.  gdb resumes every thread during an inferior call,
# so the main thread runs on through CIgManager::RunCyclic and can service the
# job -- which is exactly what deadlocked when the call was made from main.
thread 3
echo \n=== CALL ProcessJobOnAp from worker thread ===\n
print ((int (*)(void *, unsigned int, char *)) sharcCmpProcessJobOnAp)(&IgJobReq, 0, $b)
echo \n=== RETURNED ===\n
