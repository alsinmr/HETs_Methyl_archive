import sys
from chimerax.core.commands import run as rc

rc(session,"remotecontrol rest start port 60975")
sys.path.append("/Users/albertsmith/Documents/GitHub/pyDR/chimeraX")
sys.path.append("/Users/albertsmith/Documents/GitHub/pyDR")
from RemoteCMXside import CMXReceiver as CMXR
import RemoteCMXside
cmxr=CMXR(session,7017,rc_port0=60975)
rc(session,"ui mousemode right select")
