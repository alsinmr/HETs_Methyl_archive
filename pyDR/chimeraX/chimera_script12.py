import sys
from chimerax.core.commands import run as rc

rc(session,"remotecontrol rest start port 60970")
sys.path.append("/Users/albertsmith/Documents/GitHub/pyDR/chimeraX")
sys.path.append("/Users/albertsmith/Documents/GitHub/pyDR")
from RemoteCMXside import CMXReceiver as CMXR
import RemoteCMXside
cmxr=CMXR(session,7012,rc_port0=60970)
rc(session,"ui mousemode right select")
