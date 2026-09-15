import socket, time
targets = [("10.40.131.19", 45673), ("10.40.130.126", 44819), ("10.40.128.111", 44545)]
for host, port in targets:
    print("=== %s:%d ===" % (host, port))
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(6)
    t0 = time.time()
    try:
        s.connect((host, port))
        print("  TCP connect OK in %.3fs" % (time.time()-t0))
        s.settimeout(4)
        try:
            d = s.recv(64)
            print("  recv %d bytes: %r" % (len(d), d[:40]))
        except socket.timeout:
            print("  recv TIMEOUT -> peer silent (adbd not answering)")
        except Exception as e:
            print("  recv err:", type(e).__name__, str(e)[:60])
    except socket.timeout:
        print("  connect TIMEOUT (%.1fs) -> no SYN-ACK" % (time.time()-t0))
    except Exception as e:
        print("  connect err:", type(e).__name__, str(e)[:60])
    finally:
        s.close()
