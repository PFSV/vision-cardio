import Combine
import CoreVideo
import Foundation

/// Drives the live pipeline: camera frames -> clip tensor -> Core ML HR model ->
/// CoachSignal -> CoachEngine decision, all published for the SwiftUI view.
///
/// If the model isn't bundled yet, `modelLoaded` is false and the UI shows a
/// placeholder; the rest of the app still runs.
@MainActor
final class CoachViewModel: ObservableObject {
    @Published var signal: CoachSignal
    @Published var decision: CoachDecision
    @Published var modelLoaded: Bool
    @Published var statusText: String = "대기 중"

    // 운동 모드 + 모드별 표시값
    @Published var mode: ExerciseMode = .running
    @Published var phase: Int = 1
    @Published var phaseText: String = "측정 대기"
    @Published var phaseHint: String = ""
    @Published var weightsLabel: String = "세트를 시작하면 자동으로 감지합니다"
    @Published var recoveryPct: Double = 0

    // 세션 단계 + 기준(휴식) 심박
    @Published var session: SessionState = .idle
    @Published var baselineBpm: Double = 0

    /// 최대심박(러닝 존 기준). 기본 190 ≈ 220-30. 추후 나이로 보정 가능.
    let maxBpm: Double = 190
    private var weights = WeightsTracker()
    private var calibSamples: [Double] = []
    private let calibNeeded = 5      // 기준 측정에 쓸 신뢰 측정 개수

    let camera = CameraManager()
    private let model: HeartRateModel?
    private let buffer: FrameClipBuffer
    private var timer: Timer?
    private let inferQueue = DispatchQueue(label: "coach.inference", qos: .userInitiated)
    private var inferring = false

    private let restingBpm: Double
    private let confidenceThreshold: Double = 0.4
    private var lastBpm: Double?
    // Exponential-moving-average state to calm the per-second jitter.
    private var smBpm: Double?
    private var smConf: Double?

    init(restingBpm: Double = 68) {
        self.restingBpm = restingBpm
        let m = HeartRateModel()
        self.model = m
        self.modelLoaded = (m != nil)
        self.buffer = FrameClipBuffer(clipFrames: m?.clipFrames ?? 128,
                                      clipSize: m?.clipSize ?? 112,
                                      clipSeconds: m?.clipSeconds ?? 20.0)
        self.signal = CoachSignal(currentHrBpm: restingBpm, restingHrBpm: restingBpm,
                                  confidence: 0, recoveryScore: 0.7, trendBpm: 0,
                                  completedLastSession: true, faceVisible: false,
                                  ambientLightOK: true)
        self.decision = CoachDecision(recommendation: "defer", intensityStep: 0,
                                      reason: "카메라 신호 대기 중")
        camera.onFrame = { [weak self] pb in self?.buffer.append(pixelBuffer: pb) }
    }

    func start() {
        camera.requestAccess()
        camera.start()
        session = .calibrating
        calibSamples = []; baselineBpm = 0
        weights = WeightsTracker(); recoveryPct = 0
        statusText = modelLoaded ? "프레임 수집 중…" : "모델 미탑재 — VisionCardioHR.mlpackage 추가 필요"
        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            self?.tick()
        }
    }

    func stop() {
        camera.stop()
        timer?.invalidate(); timer = nil
        buffer.reset()
        smBpm = nil; smConf = nil; lastBpm = nil
        weights = WeightsTracker(); recoveryPct = 0
        session = .idle; calibSamples = []; baselineBpm = 0
        phaseText = "측정 대기"; phaseHint = ""
        weightsLabel = "세트를 시작하면 자동으로 감지합니다"
        statusText = "정지됨"
    }

    /// Once per second: if a full clip is buffered and the model is loaded, run a
    /// prediction off the main thread and fold the result into the published signal.
    private func tick() {
        guard let model, modelLoaded else { return }
        guard buffer.isReady else {
            statusText = String(format: "측정 준비 중… %.0f%%", buffer.fillRatio * 100)
            return
        }
        guard !inferring else { return }
        guard let clip = buffer.makeClip() else { return }
        inferring = true
        inferQueue.async { [weak self] in
            let est = model.predict(clip: clip)
            DispatchQueue.main.async {
                self?.inferring = false
                guard let self, let est else { return }
                self.apply(estimate: est)
            }
        }
    }

    private func apply(estimate est: HeartRateModel.Estimate) {
        // Smooth HR (only when confident — a low-confidence window shouldn't yank the
        // displayed HR) and smooth confidence so the badge stops flickering.
        let aConf = 0.3
        let conf = smConf.map { $0 * (1 - aConf) + est.confidence * aConf } ?? est.confidence
        smConf = conf

        let bpm: Double
        if est.confidence >= confidenceThreshold {
            let aBpm = 0.35
            bpm = smBpm.map { $0 * (1 - aBpm) + est.bpm * aBpm } ?? est.bpm
        } else {
            bpm = smBpm ?? est.bpm   // hold last good HR through a noisy window
        }
        smBpm = bpm

        // 추세(직전 측정 대비 변화) — 웨이트의 쉴때/안쉴때 판단에 사용.
        let trend = lastBpm.map { bpm - $0 } ?? 0
        lastBpm = bpm

        // 신뢰할 만한 측정일 때만: 먼저 기준 심박을 모으고(calibrating) → 코칭(active).
        if est.confidence >= confidenceThreshold {
            switch session {
            case .calibrating:
                calibSamples.append(bpm)
                if calibSamples.count >= calibNeeded {
                    baselineBpm = calibSamples.sorted()[calibSamples.count / 2]   // 중앙값
                    session = .active
                }
            case .active:
                let z = RunCoach.zone(bpm: bpm, maxBpm: maxBpm)
                phase = z.phase
                phaseText = "페이즈 \(z.phase) · \(z.name)"
                phaseHint = z.hint
                weights.update(bpm: bpm, baseline: baselineBpm, trend: trend)
                weightsLabel = weights.label(baseline: baselineBpm)
                recoveryPct = weights.recoveryPct
            case .idle:
                break
            }
        }

        let recovery = max(0, min(1, 1 - abs(trend) / 20))
        let faceVisible = conf >= confidenceThreshold

        signal = CoachSignal(
            currentHrBpm: bpm,
            restingHrBpm: restingBpm,
            confidence: conf,
            recoveryScore: recovery,
            trendBpm: trend,
            completedLastSession: signal.completedLastSession,
            faceVisible: faceVisible,
            ambientLightOK: true
        )
        decision = CoachEngine.decide(signal, baselineBpm: restingBpm,
                                      confidenceThreshold: confidenceThreshold)
        if session == .calibrating {
            statusText = "기준 심박 측정 중… \(calibSamples.count)/\(calibNeeded) · 가만히 계세요"
        } else {
            statusText = String(format: "심박 %.0f bpm · 신뢰도 %.0f%%", bpm, conf * 100)
        }
    }
}
