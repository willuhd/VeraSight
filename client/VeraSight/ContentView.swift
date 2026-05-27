//
//  ContentView.swift
//  VeraSight
//
//  Created by Will on 5/23/26.
//

import SwiftUI
import ARKit
import Network
import Compression
import Combine
import MetalKit

struct FaceMeshData {
    let vertices: [simd_float3]
    let leftEye: simd_float3
    let rightEye: simd_float3
    let lookAt: simd_float3
}

// MARK: - Metal Renderer & Shaders
class MetalRenderer: NSObject, MTKViewDelegate {
    var device: MTLDevice!
    var commandQueue: MTLCommandQueue!
    var pipelineStateFace: MTLRenderPipelineState?
    var pipelineStateGazeLine: MTLRenderPipelineState?
    var pipelineStatePupil: MTLRenderPipelineState?
    
    private var meshData: FaceMeshData?
    private var isDarkMode: Bool = false
    
    private var vertexBuffer: MTLBuffer?
    private var gazeBuffer: MTLBuffer?
    
    struct Uniforms {
        var minZ: Float
        var maxZ: Float
        var isDarkMode: UInt32
    }
    
    init(device: MTLDevice) {
        self.device = device
        self.commandQueue = device.makeCommandQueue()
        super.init()
        setupPipelines()
    }
    
    private func setupPipelines() {
        let shaderSource = """
        #include <metal_stdlib>
        using namespace metal;

        struct VertexOut {
            float4 position [[position]];
            float point_size [[point_size]];
            float4 color;
        };

        struct Uniforms {
            float minZ;
            float maxZ;
            uint isDarkMode;
        };

        vertex VertexOut face_vertex(const device float3* vertices [[buffer(0)]],
                                     constant Uniforms& uniforms [[buffer(1)]],
                                     uint vid [[vertex_id]]) {
            VertexOut out;
            float3 v = vertices[vid];
            
            // Map X and Y coordinates from [-0.1, 0.1] to NDC [-1.0, 1.0] with padding
            out.position = float4(v.x / 0.12, v.y / 0.12, 0.0, 1.0);
            out.point_size = 4.0;
            
            float normZ = (v.z - uniforms.minZ) / (uniforms.maxZ - uniforms.minZ);
            float intensity = clamp(normZ, 0.0, 1.0);
            
            if (uniforms.isDarkMode != 0) {
                out.color = float4(intensity * 0.1, intensity * 0.7, intensity * 1.0, 1.0);
            } else {
                float adjusted = 0.35 + (intensity * 0.55);
                out.color = float4(adjusted * 0.1, adjusted * 0.7, adjusted * 1.0, 1.0);
            }
            return out;
        }

        fragment float4 face_fragment(VertexOut in [[stage_in]]) {
            return in.color;
        }

        struct GazeVertexOut {
            float4 position [[position]];
            float point_size [[point_size]];
        };

        vertex GazeVertexOut gaze_vertex(const device float3* positions [[buffer(0)]],
                                         uint vid [[vertex_id]]) {
            GazeVertexOut out;
            float3 v = positions[vid];
            out.position = float4(v.x / 0.12, v.y / 0.12, 0.0, 1.0);
            out.point_size = 8.0;
            return out;
        }

        fragment float4 gaze_line_fragment(GazeVertexOut in [[stage_in]]) {
            // Screen-space dashing pattern on the GPU
            float pattern = in.position.x + in.position.y;
            if (fmod(pattern, 12.0) > 6.0) {
                discard_fragment();
            }
            return float4(1.0, 0.5, 0.0, 1.0); // Solid standard Orange
        }

        fragment float4 pupil_fragment(GazeVertexOut in [[stage_in]]) {
            return float4(1.0, 0.5, 0.0, 1.0); // Solid pupil markers
        }
        """
        
        do {
            let library = try device.makeLibrary(source: shaderSource, options: nil)
            
            let descFace = MTLRenderPipelineDescriptor()
            descFace.vertexFunction = library.makeFunction(name: "face_vertex")
            descFace.fragmentFunction = library.makeFunction(name: "face_fragment")
            descFace.colorAttachments[0].pixelFormat = .bgra8Unorm
            descFace.colorAttachments[0].isBlendingEnabled = true
            descFace.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
            descFace.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
            
            let descGaze = MTLRenderPipelineDescriptor()
            descGaze.vertexFunction = library.makeFunction(name: "gaze_vertex")
            descGaze.fragmentFunction = library.makeFunction(name: "gaze_line_fragment")
            descGaze.colorAttachments[0].pixelFormat = .bgra8Unorm
            descGaze.colorAttachments[0].isBlendingEnabled = true
            descGaze.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
            descGaze.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
            
            let descPupil = MTLRenderPipelineDescriptor()
            descPupil.vertexFunction = library.makeFunction(name: "gaze_vertex")
            descPupil.fragmentFunction = library.makeFunction(name: "pupil_fragment")
            descPupil.colorAttachments[0].pixelFormat = .bgra8Unorm
            descPupil.colorAttachments[0].isBlendingEnabled = true
            descPupil.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
            descPupil.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
            
            pipelineStateFace = try device.makeRenderPipelineState(descriptor: descFace)
            pipelineStateGazeLine = try device.makeRenderPipelineState(descriptor: descGaze)
            pipelineStatePupil = try device.makeRenderPipelineState(descriptor: descPupil)
        } catch {
            print("Error preparing Metal shaders: \(error)")
        }
    }
    
    func updateData(meshData: FaceMeshData, isDarkMode: Bool) {
        self.meshData = meshData
        self.isDarkMode = isDarkMode
        
        guard !meshData.vertices.isEmpty else { return }
        
        let vertexSize = meshData.vertices.count * MemoryLayout<simd_float3>.stride
        if vertexBuffer == nil || vertexBuffer?.length ?? 0 < vertexSize {
            vertexBuffer = device.makeBuffer(length: vertexSize, options: .storageModeShared)
        }
        if let buf = vertexBuffer {
            memcpy(buf.contents(), meshData.vertices, vertexSize)
        }
        
        let gazePoints: [simd_float3] = [
            meshData.leftEye, meshData.lookAt,
            meshData.rightEye, meshData.lookAt,
            meshData.leftEye, meshData.rightEye
        ]
        let gazeSize = gazePoints.count * MemoryLayout<simd_float3>.stride
        if gazeBuffer == nil || gazeBuffer?.length ?? 0 < gazeSize {
            gazeBuffer = device.makeBuffer(length: gazeSize, options: .storageModeShared)
        }
        if let buf = gazeBuffer {
            gazePoints.withUnsafeBytes { ptr in
                memcpy(buf.contents(), ptr.baseAddress, gazeSize)
            }
        }
    }
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
    
    func draw(in view: MTKView) {
        guard let meshData = meshData, !meshData.vertices.isEmpty else { return }
        guard let rpd = view.currentRenderPassDescriptor,
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: rpd) else { return }
        
        if let pipeline = pipelineStateFace, let vBuf = vertexBuffer {
            encoder.setRenderPipelineState(pipeline)
            encoder.setVertexBuffer(vBuf, offset: 0, index: 0)
            
            var uniforms = Uniforms(minZ: -0.1, maxZ: 0.05, isDarkMode: isDarkMode ? 1 : 0)
            encoder.setVertexBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 1)
            
            encoder.drawPrimitives(type: .point, vertexStart: 0, vertexCount: meshData.vertices.count)
        }
        
        if let gBuf = gazeBuffer {
            if let linePipeline = pipelineStateGazeLine {
                encoder.setRenderPipelineState(linePipeline)
                encoder.setVertexBuffer(gBuf, offset: 0, index: 0)
                encoder.drawPrimitives(type: .line, vertexStart: 0, vertexCount: 4)
            }
            if let pupilPipeline = pipelineStatePupil {
                encoder.setRenderPipelineState(pupilPipeline)
                encoder.setVertexBuffer(gBuf, offset: 0, index: 0)
                encoder.drawPrimitives(type: .point, vertexStart: 4, vertexCount: 2)
            }
        }
        
        encoder.endEncoding()
        if let drawable = view.currentDrawable {
            commandBuffer.present(drawable)
        }
        commandBuffer.commit()
    }
}

struct MetalFaceMeshView: UIViewRepresentable {
    let verticesPublisher: PassthroughSubject<FaceMeshData, Never>
    @Environment(\.colorScheme) var colorScheme
    
    func makeUIView(context: Context) -> MTKView {
        let mtkView = MTKView()
        guard let defaultDevice = MTLCreateSystemDefaultDevice() else { return mtkView }
        
        mtkView.device = defaultDevice
        mtkView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        mtkView.isOpaque = false
        mtkView.framebufferOnly = true
        mtkView.enableSetNeedsDisplay = true
        
        let renderer = MetalRenderer(device: defaultDevice)
        mtkView.delegate = renderer
        
        context.coordinator.renderer = renderer
        context.coordinator.mtkView = mtkView
        context.coordinator.subscribe(to: verticesPublisher, colorScheme: colorScheme)
        
        return mtkView
    }
    
    func updateUIView(_ uiView: MTKView, context: Context) {
        context.coordinator.updateColorScheme(colorScheme)
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator()
    }
    
    class Coordinator {
        var renderer: MetalRenderer?
        var mtkView: MTKView?
        private var cancellable: AnyCancellable?
        private var lastColorScheme: ColorScheme?
        
        func subscribe(to publisher: PassthroughSubject<FaceMeshData, Never>, colorScheme: ColorScheme) {
            lastColorScheme = colorScheme
            cancellable = publisher
                .receive(on: DispatchQueue.main)
                .sink { [weak self] meshData in
                    guard let self = self, let renderer = self.renderer else { return }
                    renderer.updateData(meshData: meshData, isDarkMode: self.lastColorScheme == .dark)
                    self.mtkView?.setNeedsDisplay()
                }
        }
        
        func updateColorScheme(_ colorScheme: ColorScheme) {
            lastColorScheme = colorScheme
            mtkView?.setNeedsDisplay()
        }
    }
}

// MARK: - Networking
class VSNet: ObservableObject {
    @Published var status = "Ready"
    @Published var connected = false
    @Published var framesSent = 0
    @Published var lastError = ""
    @Published var isReconnecting = false
    
    private var conn: NWConnection?
    private let q = DispatchQueue(label: "n")
    
    private var targetIP: String?
    private var isExplicitlyDisconnected = false
    private var reconnectWorkItem: DispatchWorkItem?

    func connect(_ ip: String) {
        targetIP = ip
        isExplicitlyDisconnected = false
        status = "Connecting to \(ip)..."
        
        reconnectWorkItem?.cancel()
        reconnectWorkItem = nil
        
        conn?.stateUpdateHandler = nil
        conn?.cancel()
        
        let ep = NWEndpoint.hostPort(
            host: NWEndpoint.Host(ip),
            port: NWEndpoint.Port(integerLiteral: 8001)
        )
        let p = NWParameters.tcp
        p.allowLocalEndpointReuse = true
        
        let newConn = NWConnection(to: ep, using: p)
        newConn.stateUpdateHandler = { [weak self] s in
            DispatchQueue.main.async {
                guard let self = self else { return }
                switch s {
                case .setup: self.status = "Setup..."
                case .waiting(_):
                    self.status = "Waiting..."
                    self.connected = false
                    self.scheduleReconnection()
                case .preparing: self.status = "Preparing..."
                case .ready:
                    self.status = "TCP Connected!"
                    self.connected = true
                    self.isReconnecting = false
                    self.lastError = ""
                    self.reconnectWorkItem?.cancel()
                    self.reconnectWorkItem = nil
                case .failed(let err):
                    self.status = "Failed: \(err)"
                    self.connected = false
                    self.lastError = "\(err)"
                    self.scheduleReconnection()
                case .cancelled:
                    self.status = "Disconnected"
                    self.connected = false
                @unknown default: self.status = "State: \(s)"
                }
            }
        }
        conn = newConn
        conn?.start(queue: q)
    }

    func disconnect() {
        isExplicitlyDisconnected = true
        isReconnecting = false
        reconnectWorkItem?.cancel()
        reconnectWorkItem = nil
        
        conn?.stateUpdateHandler = nil
        conn?.cancel()
        conn = nil
        connected = false
        status = "Disconnected"
        framesSent = 0
        lastError = ""
    }

    private func scheduleReconnection() {
        guard !isExplicitlyDisconnected, let ip = targetIP else { return }
        guard reconnectWorkItem == nil else { return }
        
        isReconnecting = true
        status = "Reconnecting..."
        
        let workItem = DispatchWorkItem { [weak self] in
            guard let self = self, !self.isExplicitlyDisconnected else { return }
            self.reconnectWorkItem = nil
            self.connect(ip)
        }
        
        reconnectWorkItem = workItem
        DispatchQueue.main.asyncAfter(deadline: .now() + 5.0, execute: workItem)
    }

    func sendBinary(_ packet: Data) {
        let length = UInt32(packet.count).bigEndian
        let prefix = withUnsafeBytes(of: length) { Data($0) }
        let payload = prefix + packet

        conn?.send(content: payload, completion: .contentProcessed { [weak self] error in
            if let err = error {
                DispatchQueue.main.async {
                    if self?.conn != nil {
                        self?.lastError = "Send err: \(err)"
                    }
                }
            } else {
                DispatchQueue.main.async {
                    self?.framesSent += 1
                }
            }
        })
    }
}

// MARK: - Capture Session
class VSCap: NSObject, ObservableObject, ARSessionDelegate {
    @Published var running = false
    @Published var debugMsg = "Stopped"
    @Published var fps = "0"
    @Published var shapes: [(String, Float)] = []
    @Published var faceDetected = false
    
    let meshPublisher = PassthroughSubject<FaceMeshData, Never>()
    
    private var ar: ARSession?
    private var ticks: [TimeInterval] = []
    private var net: VSNet?
    private var lastNet: TimeInterval = 0
    private var lastUIUpdate: TimeInterval = 0
    private var lastFaceUpdate: TimeInterval = 0
    private var sortedBlendKeys: [ARFaceAnchor.BlendShapeLocation] = []
    private var watchdogTimer: Timer?

    func bind(_ n: VSNet) { net = n }

    func start() {
        debugMsg = "Checking TrueDepth..."
        guard ARFaceTrackingConfiguration.isSupported else {
            debugMsg = "ERROR: No TrueDepth (need iPhone X or newer)"; return
        }
        
        debugMsg = "Starting camera..."
        DispatchQueue.main.async {
            withAnimation(.easeInOut(duration: 0.35)) {
                self.running = true
            }
        }
        
        ar = ARSession()
        ar?.delegate = self
        let c = ARFaceTrackingConfiguration()
        c.isLightEstimationEnabled = false
        
        if let highFPS = ARFaceTrackingConfiguration.supportedVideoFormats.first(where: { $0.framesPerSecond == 60 }) {
            c.videoFormat = highFPS
        }
        
        ar?.run(c, options: [.resetTracking, .removeExistingAnchors])
        debugMsg = "Camera ON — point at face"
        
        lastFaceUpdate = CACurrentMediaTime()
        DispatchQueue.main.async { [weak self] in
            self?.watchdogTimer = Timer.scheduledTimer(withTimeInterval: 0.15, repeats: true) { _ in
                self?.checkWatchdog()
            }
        }
    }

    func stop() {
        DispatchQueue.main.async {
            withAnimation(.easeInOut(duration: 0.35)) {
                self.running = false
                self.faceDetected = false
                self.shapes.removeAll()
            }
        }
        
        debugMsg = "Stopped"
        ar?.pause()
        ar = nil
        ticks.removeAll()
        meshPublisher.send(FaceMeshData(vertices: [], leftEye: .zero, rightEye: .zero, lookAt: .zero))
        
        DispatchQueue.main.async { [weak self] in
            self?.watchdogTimer?.invalidate()
            self?.watchdogTimer = nil
        }
    }
    
    private func checkWatchdog() {
        let now = CACurrentMediaTime()
        if now - lastFaceUpdate > 0.30 {
            if faceDetected {
                DispatchQueue.main.async { [weak self] in
                    withAnimation(.easeInOut(duration: 0.3)) {
                        self?.faceDetected = false
                        self?.shapes.removeAll()
                    }
                    self?.meshPublisher.send(FaceMeshData(vertices: [], leftEye: .zero, rightEye: .zero, lookAt: .zero))
                }
            }
        }
    }

    func session(_ s: ARSession, didUpdate anchors: [ARAnchor]) {
        guard let f = anchors.compactMap({ $0 as? ARFaceAnchor }).first else {
            DispatchQueue.main.async { [weak self] in
                guard let self = self else { return }
                if self.running {
                    self.debugMsg = "Looking for face..."
                    if self.faceDetected {
                        withAnimation(.easeInOut(duration: 0.3)) {
                            self.faceDetected = false
                            self.shapes.removeAll()
                        }
                        self.meshPublisher.send(FaceMeshData(vertices: [], leftEye: .zero, rightEye: .zero, lookAt: .zero))
                    }
                }
            }
            return
        }

        let now = CACurrentMediaTime()
        lastFaceUpdate = now
        ticks.append(now)
        ticks = ticks.filter { now - $0 < 1 }

        if sortedBlendKeys.isEmpty {
            sortedBlendKeys = f.blendShapes.keys.sorted { $0.rawValue < $1.rawValue }
        }

        let vertices = f.geometry.vertices
        
        let leftEye = simd_float3(f.leftEyeTransform.columns.3.x, f.leftEyeTransform.columns.3.y, f.leftEyeTransform.columns.3.z)
        let rightEye = simd_float3(f.rightEyeTransform.columns.3.x, f.rightEyeTransform.columns.3.y, f.rightEyeTransform.columns.3.z)
        let lookAt = f.lookAtPoint
        
        meshPublisher.send(FaceMeshData(vertices: vertices, leftEye: leftEye, rightEye: rightEye, lookAt: lookAt))

        let nowUI = CACurrentMediaTime()
        if nowUI - lastUIUpdate >= 0.066 {
            lastUIUpdate = nowUI
            let previewMap = f.blendShapes.map { ($0.key.rawValue.replacingOccurrences(of: "com.apple.", with: ""), $0.value.floatValue) }
            
            DispatchQueue.main.async { [weak self] in
                guard let self = self else { return }
                self.fps = "\(self.ticks.count)"
                
                if !self.faceDetected {
                    withAnimation(.easeInOut(duration: 0.35)) {
                        self.faceDetected = true
                        self.shapes = Array(previewMap.sorted { $0.1 > $1.1 }.prefix(5))
                    }
                } else {
                    self.shapes = Array(previewMap.sorted { $0.1 > $1.1 }.prefix(5))
                }
                self.debugMsg = "Face tracked ✓ Sending..."
            }
        }

        guard now - lastNet >= 0.016 else { return }
        lastNet = now

        var rawPayload = Data()

        func appendVal<T>(_ val: T) {
            withUnsafeBytes(of: val) { rawPayload.append(contentsOf: $0) }
        }

        func appendMatrixHalf(_ m: simd_float4x4) {
            let mArray: [Float16] = [
                Float16(m.columns.0.x), Float16(m.columns.0.y), Float16(m.columns.0.z), Float16(m.columns.0.w),
                Float16(m.columns.1.x), Float16(m.columns.1.y), Float16(m.columns.1.z), Float16(m.columns.1.w),
                Float16(m.columns.2.x), Float16(m.columns.2.y), Float16(m.columns.2.z), Float16(m.columns.2.w),
                Float16(m.columns.3.x), Float16(m.columns.3.y), Float16(m.columns.3.z), Float16(m.columns.3.w)
            ]
            mArray.withUnsafeBytes { rawPayload.append(contentsOf: $0) }
        }

        appendVal(now)
        appendMatrixHalf(f.transform)
        appendMatrixHalf(f.leftEyeTransform)
        appendMatrixHalf(f.rightEyeTransform)

        appendVal(Float16(f.lookAtPoint.x))
        appendVal(Float16(f.lookAtPoint.y))
        appendVal(Float16(f.lookAtPoint.z))

        appendVal(UInt16(sortedBlendKeys.count))
        for key in sortedBlendKeys {
            let val = f.blendShapes[key]?.floatValue ?? 0.0
            appendVal(Float16(val))
        }

        appendVal(UInt16(vertices.count))
        var flatVertices = [Float16]()
        flatVertices.reserveCapacity(vertices.count * 3)
        for v in vertices {
            flatVertices.append(Float16(v.x))
            flatVertices.append(Float16(v.y))
            flatVertices.append(Float16(v.z))
        }
        flatVertices.withUnsafeBytes { rawPayload.append(contentsOf: $0) }

        var finalPacket = Data()
        finalPacket.append(contentsOf: [0x56, 0x53, 0x42, 0x50])
        
        do {
            if let compressed = try (rawPayload as NSData).compressed(using: .zlib) as Data? {
                finalPacket.append(compressed)
            } else {
                finalPacket.append(rawPayload)
            }
        } catch {
            finalPacket.append(rawPayload)
        }

        net?.sendBinary(finalPacket)
    }
}

// MARK: - Views
struct ContentView: View {
    @StateObject private var net = VSNet()
    @StateObject private var cap = VSCap()
    @State private var ip = "192.168.3.174"

    private var indicatorText: String {
        if !cap.running {
            return "Stopped"
        }
        if !cap.faceDetected {
            return "Searching"
        }
        return "FPS: \(cap.fps)"
    }

    private var indicatorColor: Color {
        if !cap.running {
            return .gray
        }
        if !cap.faceDetected {
            return .yellow
        }
        return net.connected ? .green : .orange
    }
    
    @ViewBuilder
    private var targetCompanionCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Target Host")
                .font(.footnote)
                .fontWeight(.semibold)
                .foregroundColor(.secondary)
            
            HStack(spacing: 12) {
                Image(systemName: "laptopcomputer")
                    .foregroundColor(.accentColor)
                    .font(.body)
                
                Text("Server IP")
                    .font(.body)
                
                Spacer()
                
                TextField("192.168.3.174", text: $ip)
                    .keyboardType(.decimalPad)
                    .font(.system(.body, design: .monospaced))
                    .multilineTextAlignment(.trailing)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 170)
                    .disabled(cap.running)
                    .opacity(cap.running ? 0.6 : 1.0)
            }
            
            if net.isReconnecting {
                VStack(spacing: 0) {
                    Divider().padding(.vertical, 8)
                    HStack(spacing: 6) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundColor(.orange)
                            .font(.caption)
                        Text("Reconnecting...")
                            .font(.caption)
                            .foregroundColor(.orange)
                        Spacer()
                    }
                }
                .transition(.opacity)
            } else if net.connected && !net.lastError.isEmpty {
                VStack(spacing: 0) {
                    Divider().padding(.vertical, 8)
                    HStack(spacing: 6) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundColor(.red)
                            .font(.caption)
                        Text(net.lastError)
                            .font(.caption)
                            .foregroundColor(.red)
                        Spacer()
                    }
                }
                .transition(.opacity)
            }
        }
        .padding()
        .background(Color(.secondarySystemGroupedBackground))
        .cornerRadius(12)
        .padding(.horizontal)
    }

    @ViewBuilder
    private var idleView: some View {
        VStack(spacing: 16) {
            Spacer()
            Image(systemName: "face.dashed")
                .font(.system(size: 56))
                .foregroundColor(.secondary)
            Text("VeraSight is Idle")
                .font(.title3)
                .fontWeight(.bold)
            Text("Enter your target host's IP address above and tap Start to begin mapping.")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 32)
            Spacer()
        }
        .frame(minHeight: 340)
    }

    @ViewBuilder
    private var scanningView: some View {
        VStack(spacing: 16) {
            Spacer()
            ProgressView()
                .scaleEffect(1.2)
                .padding(.bottom, 4)
            Image(systemName: "person.fill.viewfinder")
                .font(.system(size: 56))
                .foregroundColor(.orange)
            Text("Scanning for Face")
                .font(.title3)
                .fontWeight(.bold)
            Text("Ensure your face is clearly visible to the front-facing camera.")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 32)
            Spacer()
        }
        .frame(minHeight: 340)
    }

    @ViewBuilder
    private var expressionsCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Blend Shapes")
                .font(.footnote)
                .fontWeight(.semibold)
                .foregroundColor(.secondary)
            
            VStack(spacing: 10) {
                ForEach(cap.shapes.indices, id: \.self) { index in
                    let item = cap.shapes[index]
                    HStack {
                        Text(item.0)
                            .font(.caption2)
                            .lineLimit(1)
                            .frame(width: 100, alignment: .leading)
                        
                        GeometryReader { g in
                            ZStack(alignment: .leading) {
                                Capsule()
                                    .fill(Color.gray.opacity(0.2))
                                    .frame(height: 8)
                                Capsule()
                                    .fill(
                                        LinearGradient(
                                            colors: [Color.blue, Color.cyan],
                                            startPoint: .leading,
                                            endPoint: .trailing
                                        )
                                    )
                                    .frame(width: max(1, CGFloat(item.1) * g.size.width), height: 8)
                                    .animation(.spring(response: 0.18, dampingFraction: 0.8), value: item.1)
                            }
                        }
                        .frame(height: 8)
                        
                        Text(String(format: "%.2f", item.1))
                            .font(.caption2)
                            .frame(width: 32, alignment: .trailing)
                    }
                }
            }
        }
        .padding()
        .background(Color(.secondarySystemGroupedBackground))
        .cornerRadius(12)
        .padding(.horizontal)
    }

    @ViewBuilder
    private var meshCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("3D Face & Gaze Tracking")
                .font(.footnote)
                .fontWeight(.semibold)
                .foregroundColor(.secondary)
            
            HStack {
                Spacer()
                MetalFaceMeshView(verticesPublisher: cap.meshPublisher)
                    .aspectRatio(1.0, contentMode: .fit)
                    .frame(maxWidth: 280)
                Spacer()
            }
        }
        .padding()
        .background(Color(.secondarySystemGroupedBackground))
        .cornerRadius(12)
        .padding(.horizontal)
    }

    @ViewBuilder
    private var bottomActionBar: some View {
        VStack(spacing: 0) {
            Divider()
            Button {
                withAnimation(.easeInOut(duration: 0.35)) {
                    if cap.running {
                        cap.stop()
                        net.disconnect()
                    } else {
                        cap.bind(net)
                        net.connect(ip)
                        cap.start()
                    }
                }
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: cap.running ? "stop.fill" : "play.fill")
                    Text(cap.running ? "Stop Tracking" : "Start Tracking")
                }
                .font(.headline)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 14)
                .background(cap.running ? Color.red : Color.accentColor)
                .foregroundColor(.white)
                .cornerRadius(12)
            }
            .padding([.horizontal, .top])
            .padding(.bottom, 16)
        }
        .background(Color(.systemBackground))
    }

    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                ScrollView {
                    VStack(spacing: 16) {
                        targetCompanionCard
                        
                        if !cap.running {
                            idleView
                                .transition(.opacity.combined(with: .scale(scale: 0.95)))
                        } else if !cap.faceDetected {
                            scanningView
                                .transition(.opacity.combined(with: .scale(scale: 0.95)))
                        } else {
                            expressionsCard
                                .transition(.opacity.combined(with: .scale(scale: 0.95)))
                            meshCard
                                .transition(.opacity.combined(with: .scale(scale: 0.95)))
                        }
                    }
                    .padding(.vertical)
                    .animation(.easeInOut(duration: 0.35), value: cap.running)
                    .animation(.easeInOut(duration: 0.35), value: cap.faceDetected)
                    .animation(.easeInOut(duration: 0.35), value: net.lastError)
                    .animation(.easeInOut(duration: 0.35), value: net.isReconnecting)
                }
                .scrollDismissesKeyboard(.immediately)
                .background(Color(.systemGroupedBackground).ignoresSafeArea())

                bottomActionBar
            }
            .navigationTitle("VeraSight")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    HStack(spacing: 6) {
                        Circle()
                            .fill(indicatorColor)
                            .frame(width: 8, height: 8)
                        Text(indicatorText)
                            .font(.subheadline)
                            .fontWeight(.semibold)
                            .foregroundColor(.primary)
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 4)
                    .background(Color(.systemGroupedBackground))
                    .clipShape(Capsule())
                }
            }
        }
        .navigationViewStyle(.stack) // Resolves landscape split-view sidebar formatting
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
