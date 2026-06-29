import CoreML
import Foundation

/// Loads the exported Core ML rPPG model (`VisionCardioHR.mlpackage`) and turns a
/// clip of camera frames into a heart-rate estimate.
///
/// Contract must match `ml/physnet.py` + `ml/export_coreml.py` (PhysNet, the model
/// that replaced the abandoned HR-bin classifier — see notes/POSTMORTEM_rppg_3dconv.md):
///   - input  "clip":     MLMultiArray [1, 3, T, H, W], float 0...1 (RGB, normalized)
///   - output "waveform": MLMultiArray [1, T]  (predicted rPPG pulse waveform)
///   - HR = FFT peak of the waveform within [loHz, hiHz], fs = clipFrames / clipSeconds
final class HeartRateModel {
    struct Estimate {
        let bpm: Double
        let confidence: Double   // spectral peak prominence in the HR band, 0...1
    }

    let clipFrames: Int
    let clipSize: Int
    let clipSeconds: Double      // real-time span the clip should cover (training window)
    let fs: Double               // waveform sampling rate (Hz) = clipFrames / clipSeconds
    private let loHz: Double
    private let hiHz: Double
    private let model: MLModel

    /// Looks up `VisionCardioHR` in the app bundle. Returns nil if the model
    /// hasn't been added to the target yet (so the app still builds/runs without it).
    init?(bundle: Bundle = .main) {
        guard let url = bundle.url(forResource: "VisionCardioHR", withExtension: "mlmodelc")
            ?? bundle.url(forResource: "VisionCardioHR", withExtension: "mlpackage") else {
            return nil
        }
        let config = MLModelConfiguration()
        // PhysNet is a 3D-conv (Conv3d/MaxPool3d) model; the Neural Engine can crash
        // on these 5D ops on-device (uncatchable low-level fault). GPU/CPU is reliable.
        config.computeUnits = .cpuAndGPU
        guard let loaded = try? MLModel(contentsOf: url, configuration: config) else {
            return nil
        }
        self.model = loaded

        // Metadata written by ml/export_coreml.py: clip_frames=128, clip_seconds=20.0,
        // clip_size=112, hr_band_hz="0.7,3.0".
        let md = loaded.modelDescription.metadata[.creatorDefinedKey] as? [String: String] ?? [:]
        let frames = Int(md["clip_frames"] ?? "") ?? 128
        let seconds = Double(md["clip_seconds"] ?? "") ?? 20.0
        self.clipFrames = frames
        self.clipSize = Int(md["clip_size"] ?? "") ?? 112
        self.clipSeconds = seconds > 0 ? seconds : 20.0
        self.fs = seconds > 0 ? Double(frames) / seconds : 6.4
        let band = (md["hr_band_hz"] ?? "0.7,3.0").split(separator: ",").compactMap { Double($0) }
        self.loHz = band.count == 2 ? band[0] : 0.7
        self.hiHz = band.count == 2 ? band[1] : 3.0
    }

    /// `clip` must be [1, 3, clipFrames, clipSize, clipSize], float 0...1.
    func predict(clip: MLMultiArray) -> Estimate? {
        let input = try? MLDictionaryFeatureProvider(dictionary: ["clip": clip])
        guard let input, let out = try? model.prediction(from: input),
              let wave = out.featureValue(for: "waveform")?.multiArrayValue else {
            return nil
        }
        let n = wave.count
        guard n >= 4 else { return nil }
        var w = [Double](repeating: 0, count: n)
        for i in 0..<n { w[i] = wave[i].doubleValue }
        return Self.hrFromWave(w, fs: fs, loHz: loHz, hiHz: hiHz)
    }

    /// HR (bpm) = peak of the Hann-windowed power spectrum in [loHz, hiHz].
    /// Mirrors `ml/physnet.hr_from_wave` so on-device HR matches the training-time decode.
    /// n is small (T≈128) so a direct periodogram over the rfft bins is trivial.
    static func hrFromWave(_ wave: [Double], fs: Double, loHz: Double, hiHz: Double) -> Estimate {
        let n = wave.count
        let mean = wave.reduce(0, +) / Double(n)
        var w = [Double](repeating: 0, count: n)
        for t in 0..<n {
            let hann = 0.5 - 0.5 * cos(2.0 * Double.pi * Double(t) / Double(n - 1))
            w[t] = (wave[t] - mean) * hann
        }
        let half = n / 2
        var powers: [Double] = []
        var peakPow = -1.0
        var peakHz = 0.0
        for k in 0...half {
            let freq = Double(k) * fs / Double(n)
            if freq < loHz || freq > hiHz { continue }
            var re = 0.0, im = 0.0
            let ang = -2.0 * Double.pi * Double(k) / Double(n)
            for t in 0..<n {
                let a = ang * Double(t)
                re += w[t] * cos(a)
                im += w[t] * sin(a)
            }
            let p = re * re + im * im
            powers.append(p)
            if p > peakPow { peakPow = p; peakHz = freq }
        }
        let bpm = peakHz * 60.0
        // Confidence = spectral SNR: how far the peak bin stands above the band's
        // MEDIAN bin, mapped to 0…1. (peak / total reads misleadingly low — one bin
        // of ~46 is only ~2% even for a perfectly clean peak.)
        var conf = 0.0
        if peakPow > 0, !powers.isEmpty {
            let sorted = powers.sorted()
            let median = sorted[sorted.count / 2]
            let snr = median > 1e-12 ? peakPow / median : Double(sorted.count)
            conf = snr > 1 ? min(1.0, 1.0 - 1.0 / snr) : 0.0
        }
        return Estimate(bpm: bpm, confidence: conf)
    }
}
