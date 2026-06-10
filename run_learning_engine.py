#!/usr/bin/env python3
"""Standalone Learning Engine - runs as a separate service."""
import signal
import sys
import os
sys.path.insert(0, '/home/ubuntu/investwise')
os.chdir('/home/ubuntu/investwise')

from learning_engine import start_learning_engine, stop_learning_engine

engine = start_learning_engine()

def _shutdown(sig, frame):
    stop_learning_engine()
    sys.exit(0)

signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT, _shutdown)

# Keep alive
import time
while True:
    time.sleep(60)
