import socket, struct, sys, time

def main(path):
    s = socket.socket(); s.connect(('127.0.0.1', 9999))
    s.sendall(b"WATCHHAND\n"); s.sendall(struct.pack('<i', 44100)); s.sendall(bytes([1, 16]))
    data = open(path, 'rb').read()
    for i in range(0, len(data), 4410):
        s.sendall(data[i:i+4410])
        time.sleep(0.05)  # 实时速率
    s.close()
    print('replay done')

main(sys.argv[1])
