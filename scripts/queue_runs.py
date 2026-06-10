import os, sys, subprocess, time
os.chdir("/home/ubuntu/investwise")
log = open("/tmp/qlog.txt", "w")
log.write("watching saudi pid 1269185\n"); log.flush()
pid = 1269185
while True:
    try: os.kill(pid, 0); time.sleep(15)
    except ProcessLookupError: break
log.write("saudi run1 done\n"); log.flush()
p2 = subprocess.Popen(
    [sys.executable, "scripts/populate_saudi_fundamentals.py"],
    stdout=open("/tmp/saudi_pop3.log", "w"), stderr=subprocess.STDOUT)
log.write("saudi run2 pid=" + str(p2.pid) + "\n"); log.flush()
p2.wait()
log.write("saudi run2 done\n"); log.flush()
p3 = subprocess.Popen(
    [sys.executable, "scripts/populate_egypt_fundamentals.py"],
    stdout=open("/tmp/egypt_pop.log", "w"), stderr=subprocess.STDOUT)
log.write("egypt pid=" + str(p3.pid) + "\n"); log.flush()
p3.wait()
log.write("ALL DONE\n"); log.flush(); log.close()
