import socket, struct, zlib, time

def adb_packet(cmd, arg0, arg1, payload=b""):
    magic = cmd ^ 0xFFFFFFFF
    hdr = struct.pack("<6I", cmd, arg0, arg1, len(payload), zlib.crc32(payload) & 0xFFFFFFFF, magic)
    return hdr + payload

CNXN = 0x4E584E43
AUTH = 0x48545541
OKAY = 0x59414B4F
A_VERSION = 0x01000001

targets = [("10.40.131.19", 45673), ("10.40.131.19", 44819)]
payload = b"host::features=cmd,stat,ls_v2,shell_v2\x00"
for host, port in targets:
    print("=== %s:%d ===" % (host, port))
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(8)
    try:
        s.connect((host, port))
        print("  TCP OK; sending CNXN...")
        s.sendall(adb_packet(CNXN, A_VERSION, 256*1024, payload))
        s.settimeout(6)
        hdr = s.recv(24)
        if len(hdr) < 24:
            print("  short/no reply: %r" % hdr)
        else:
            cmd, a0, a1, dlen, dcrc, magic = struct.unpack("<6I", hdr)
            name = {CNXN:'CNXN', AUTH:'AUTH', OKAY:'OKAY', 0x4E584E43:'CNXN'}.get(cmd, hex(cmd))
            print("  REPLY cmd=%s arg0=0x%08x arg1=%d dlen=%d" % (name, a0, a1, dlen))
            if dlen and dlen < 4096:
                body = s.recv(min(dlen, 1024))
                print("  body[:120]=%r" % body[:120])
    except socket.timeout:
        print("  TIMEOUT waiting for adb reply -> adbd NOT serving this port")
    except Exception as e:
        print("  err:", type(e).__name__, str(e)[:70])
    finally:
        s.close()
