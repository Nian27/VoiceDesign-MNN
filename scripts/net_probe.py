import socket, time, sys
targets = [("10.40.130.126", 44819), ("192.168.0.16", 37787)]
for host, port in targets:
    print("=== %s:%d ===" % (host, port))
    for attempt in range(2):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(8)
        t0 = time.time()
        try:
            s.connect((host, port))
            dt = time.time() - t0
            print("  attempt %d: TCP connect OK in %.3fs" % (attempt, dt))
            s.settimeout(5)
            try:
                data = s.recv(64)
                print("    recv %d bytes: %r" % (len(data), data[:64]))
            except socket.timeout:
                print("    recv timeout (no banner from peer)")
            except Exception as e:
                print("    recv err:", type(e).__name__, str(e)[:80])
        except socket.timeout:
            print("  attempt %d: connect TIMEOUT after %.2fs" % (attempt, time.time()-t0))
        except Exception as e:
            print("  attempt %d: connect err: %s %s" % (attempt, type(e).__name__, str(e)[:80]))
        finally:
            s.close()
        time.sleep(1)
