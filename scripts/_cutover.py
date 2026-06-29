"""Freeze the tarball to its true contiguous prefix and print that offset."""
import os
import sys

tar = sys.argv[1]
fd = os.open(tar, os.O_RDONLY)
sz = os.fstat(fd).st_size
try:
    prefix = os.lseek(fd, 0, os.SEEK_HOLE)
except OSError:
    prefix = sz
os.close(fd)
prefix = min(prefix, sz)
os.truncate(tar, prefix)
print(prefix)
