import CoreImage
import CoreVideo
import CoreML
import Foundation
import QuartzCore

/// Rolling buffer of the most recent `clipFrames` camera frames, each resized to
/// `clipSize x clipSize` RGB. Produces the exact tensor PhysNet was trained on:
/// MLMultiArray [1, 3, T, H, W], float 0...1, RGB, channel-first — matching the
/// `scamps_pool` extraction ([0,1] normalized) that fed `ml/train_rppg.py`.
final class FrameClipBuffer {
    private let clipFrames: Int
    private let clipSize: Int
    /// Minimum real-time gap between accepted frames so that `clipFrames` frames
    /// span `clipSeconds` — matching the training window (20 s resampled to 128 f,
    /// fs 6.4 Hz). Without this the buffer holds ~4 s of 30 fps video and the model
    /// sees the wrong time scale → garbage HR + low/jittery confidence.
    private let minInterval: Double
    private var lastAccept: Double = 0
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    private let lock = NSLock()

    // Each entry is one frame as interleaved RGB floats, length 3*clipSize*clipSize.
    private var frames: [[Float]] = []
    // Scratch RGBA8 bitmap reused per frame.
    private var rgba: [UInt8]

    init(clipFrames: Int, clipSize: Int, clipSeconds: Double = 20.0) {
        self.clipFrames = clipFrames
        self.clipSize = clipSize
        self.minInterval = clipFrames > 0 ? (clipSeconds / Double(clipFrames)) * 0.95 : 0
        self.rgba = [UInt8](repeating: 0, count: clipSize * clipSize * 4)
    }

    var isReady: Bool {
        lock.lock(); defer { lock.unlock() }
        return frames.count >= clipFrames
    }

    /// 0…1 fill progress, for the "준비 중 %" status while the 20 s window loads.
    var fillRatio: Double {
        lock.lock(); defer { lock.unlock() }
        return clipFrames > 0 ? min(1.0, Double(frames.count) / Double(clipFrames)) : 1
    }

    /// Resize a camera frame to clipSize and append its RGB pixels.
    func append(pixelBuffer: CVPixelBuffer) {
        // Downsample to the training frame rate (~6.4 fps) so 128 frames cover ~20 s.
        let now = CACurrentMediaTime()
        if now - lastAccept < minInterval { return }
        lastAccept = now

        let src = CIImage(cvPixelBuffer: pixelBuffer)
        let extent = src.extent
        guard extent.width > 0, extent.height > 0 else { return }
        let scaleX = CGFloat(clipSize) / extent.width
        let scaleY = CGFloat(clipSize) / extent.height
        let scaled = src.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))

        var frame = [Float](repeating: 0, count: 3 * clipSize * clipSize)
        rgba.withUnsafeMutableBytes { raw in
            ciContext.render(scaled,
                             toBitmap: raw.baseAddress!,
                             rowBytes: clipSize * 4,
                             bounds: CGRect(x: 0, y: 0, width: clipSize, height: clipSize),
                             format: .RGBA8,
                             colorSpace: CGColorSpaceCreateDeviceRGB())
        }
        let hw = clipSize * clipSize
        let inv255: Float = 1.0 / 255.0
        for p in 0..<hw {
            frame[p] = Float(rgba[p * 4 + 0]) * inv255            // R plane, 0...1
            frame[hw + p] = Float(rgba[p * 4 + 1]) * inv255       // G plane, 0...1
            frame[2 * hw + p] = Float(rgba[p * 4 + 2]) * inv255   // B plane, 0...1
        }

        lock.lock()
        frames.append(frame)
        if frames.count > clipFrames { frames.removeFirst(frames.count - clipFrames) }
        lock.unlock()
    }

    func reset() {
        lock.lock(); frames.removeAll(keepingCapacity: true); lock.unlock()
    }

    /// Build [1, 3, T, H, W] float32. Returns nil until `clipFrames` frames are buffered.
    func makeClip() -> MLMultiArray? {
        lock.lock()
        guard frames.count >= clipFrames else { lock.unlock(); return nil }
        let snapshot = Array(frames.suffix(clipFrames))
        lock.unlock()

        let T = clipFrames, H = clipSize, W = clipSize
        guard let arr = try? MLMultiArray(shape: [1, 3, NSNumber(value: T),
                                                  NSNumber(value: H), NSNumber(value: W)],
                                          dataType: .float32) else { return nil }
        let ptr = arr.dataPointer.bindMemory(to: Float.self, capacity: arr.count)
        let hw = H * W
        let thw = T * hw
        // out[c, t, h, w] = frame[t][c*hw + (h*W + w)]
        for t in 0..<T {
            let f = snapshot[t]
            for c in 0..<3 {
                let outBase = c * thw + t * hw
                let inBase = c * hw
                for i in 0..<hw {
                    ptr[outBase + i] = f[inBase + i]
                }
            }
        }
        return arr
    }
}
