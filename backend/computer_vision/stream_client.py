import socket
import threading
import numpy as np
import cv2

class StreamClient:
    """
    Pulls JPEG frames over UDP from Unity (or any source).
    """
    def __init__(self, host:str="127.0.0.1", port:int=5005, buf_size:int=65536):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(self.addr)
        self.running = False

    def start(self, on_frame):
        """
        on_frame: callback(frame: np.ndarray)
        """
        self.running = True
        def _recv_loop():
            while self.running:
                data, _ = self.sock.recvfrom(self.buf_size)
                arr = np.frombuffer(data, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                on_frame(frame)
        threading.Thread(target=_recv_loop, daemon=True).start()

    def stop(self):
        self.running = False
        self.sock.close()