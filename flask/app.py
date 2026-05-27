import socket
import threading
import subprocess
import zlib
import struct
import json
import time
from flask import Flask, Response, render_template

# Initialize Flask with the template folder pointing to the current directory
app = Flask(__name__, template_folder='.')

def get_physical_wifi_ip():
    interfaces = ["en0", "en1"]
    for interface in interfaces:
        try:
            ip = subprocess.check_output(["ipconfig", "getifaddr", interface]).decode("utf-8").strip()
            if ip: return ip
        except subprocess.CalledProcessError: continue
    return None

# Thread-safe global state for TCP receiver and SSE pipeline
frame_cond = threading.Condition()
latest_frame_data = None
connection_status = {
    "connected": False,
    "ip": "N/A",
    "frames": 0,
    "bytes_received": 0,
    "start_time": None
}

def decompress_payload(compressed_data):
    try:
        return zlib.decompress(compressed_data)
    except zlib.error:
        return zlib.decompress(compressed_data, -15)

def recv_exactly(sock, num_bytes):
    buf = bytearray()
    while len(buf) < num_bytes:
        packet = sock.recv(num_bytes - len(buf))
        if not packet:
            return None
        buf.extend(packet)
    return bytes(buf)

def tcp_receiver():
    global latest_frame_data
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', 8001))
    server_socket.listen(1)
    
    print("VeraSight webview receiver listening on port 8001...")
    
    while True:
        conn = None
        try:
            conn, addr = server_socket.accept()
            connection_status["connected"] = True
            connection_status["ip"] = addr[0]
            connection_status["start_time"] = time.time()
            connection_status["frames"] = 0
            connection_status["bytes_received"] = 0
            
            while True:
                len_data = recv_exactly(conn, 4)
                if not len_data:
                    break
                
                msg_len = struct.unpack('>I', len_data)[0]
                connection_status["bytes_received"] += 4
                
                payload = recv_exactly(conn, msg_len)
                if not payload:
                    break
                
                connection_status["bytes_received"] += msg_len
                connection_status["frames"] += 1
                
                magic = payload[0:4]
                if magic != b'VSBP':
                    continue
                
                try:
                    decompressed = decompress_payload(payload[4:])
                    
                    header_format = "<d48e3e"
                    header_size = struct.calcsize(header_format)
                    header_data = struct.unpack(header_format, decompressed[0:header_size])
                    
                    timestamp = header_data[0]
                    head_transform = list(header_data[1:17])
                    left_eye_transform = list(header_data[17:33])
                    right_eye_transform = list(header_data[33:49])
                    look_at = list(header_data[49:52])
                    
                    idx = header_size
                    
                    blend_count = struct.unpack("<H", decompressed[idx:idx+2])[0]
                    idx += 2
                    blend_values = struct.unpack(f"<{blend_count}e", decompressed[idx:idx + (blend_count * 2)])
                    idx += blend_count * 2
                    
                    vertex_count = struct.unpack("<H", decompressed[idx:idx+2])[0]
                    idx += 2
                    flat_vertices = struct.unpack(f"<{vertex_count * 3}e", decompressed[idx:idx + (vertex_count * 3 * 2)])
                    
                    frame = {
                        "t": timestamp,
                        "la": look_at,
                        "ht": head_transform,
                        "le": left_eye_transform,
                        "re": right_eye_transform,
                        "sh": list(blend_values),
                        "vt": list(flat_vertices),
                        "meta": {
                            "connected": True,
                            "ip": connection_status["ip"],
                            "frames": connection_status["frames"],
                            "bytes_received": connection_status["bytes_received"],
                            "elapsed": time.time() - connection_status["start_time"]
                        }
                    }
                    
                    with frame_cond:
                        latest_frame_data = frame
                        frame_cond.notify_all()
                        
                except Exception as e:
                    print(f"Error parsing incoming frame packet: {e}")
                    
        except Exception as e:
            print(f"TCP socket loop exception: {e}")
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
            connection_status["connected"] = False
            connection_status["ip"] = "N/A"
            
            with frame_cond:
                latest_frame_data = {"meta": {"connected": False}}
                frame_cond.notify_all()
            print("VeraSight client disconnected")

# Start background receiver thread
receiver_thread = threading.Thread(target=tcp_receiver, daemon=True)
receiver_thread.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stream')
def stream():
    def event_stream():
        while True:
            with frame_cond:
                frame_cond.wait()
                if latest_frame_data:
                    data = json.dumps(latest_frame_data)
            yield f"data: {data}\n\n"
    return Response(event_stream(), mimetype="text/event-stream")

if __name__ == '__main__':
    wifi_ip = get_physical_wifi_ip()
    if wifi_ip: print("Enter " + wifi_ip + " on VeraSight client")
    app.run(host='0.0.0.0', port=8765, threaded=True)