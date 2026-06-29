import Foundation

/// 운동 모드 — 러닝(HR 존/페이즈)과 웨이트(세트/휴식 회복).
enum ExerciseMode: String, CaseIterable, Identifiable {
    case running = "러닝"
    case weights = "웨이트"
    var id: String { rawValue }
}

/// 러닝 HR 존(페이즈 1~5). %maxHR 기준.
struct RunZone {
    let phase: Int
    let name: String
    let hint: String
}

enum RunCoach {
    /// 현재 심박과 최대심박으로 페이즈 산출.
    static func zone(bpm: Double, maxBpm: Double) -> RunZone {
        let pct = maxBpm > 0 ? bpm / maxBpm : 0
        switch pct {
        case ..<0.60: return RunZone(phase: 1, name: "회복", hint: "아주 가벼움 · 워밍업/쿨다운")
        case ..<0.70: return RunZone(phase: 2, name: "유산소 기초", hint: "대화 가능한 페이스 · 지구력")
        case ..<0.80: return RunZone(phase: 3, name: "유산소", hint: "약간 숨참 · 심폐 향상")
        case ..<0.90: return RunZone(phase: 4, name: "젖산 역치", hint: "힘듦 · 짧게 반복")
        default:      return RunZone(phase: 5, name: "최대", hint: "전력 · 아주 짧게만")
        }
    }
}

/// 세션 단계: 시작하면 먼저 기준 심박을 재고(calibrating) → 코칭(active).
enum SessionState { case idle, calibrating, active }

/// 웨이트 세트/휴식 상태 머신. **명시적으로 측정한 기준(휴식) 심박** 대비 현재 심박으로
/// "지금 쉬어야 하는지 / 쉰다면 다시 할 타이밍인지"를 판단.
struct WeightsTracker {
    enum State { case idle, working, recovering, ready }

    private(set) var state: State = .idle
    private var setPeak: Double = 0       // 이번 세트 최고 심박
    private let workMargin = 15.0         // 기준 +15 넘으면 "운동 중"으로 판단
    private let readyMargin = 8.0         // 기준 +8 이내로 내려오면 "회복 완료"
    private(set) var recoveryPct: Double = 0

    /// `trend` = 직전 측정 대비 심박 변화(bpm). 양수=상승(운동), 음수=하강(휴식).
    /// 쉴 때/안 쉴 때를 이 추세로 판단한다.
    mutating func update(bpm: Double, baseline: Double, trend: Double) {
        switch state {
        case .idle, .ready:
            if bpm > baseline + workMargin || trend > 2 {   // 급등 = 세트 시작
                state = .working; setPeak = bpm; recoveryPct = 0
            }
        case .working:
            setPeak = max(setPeak, bpm)
            if trend < -0.5, bpm < setPeak - 3 { state = .recovering }  // 추세 하강 → 휴식 진입
        case .recovering:
            if trend > 1.5, bpm > baseline + workMargin {    // 다시 상승 → 세트 재개
                state = .working; setPeak = bpm
            } else {
                let span = max(1, setPeak - baseline)
                recoveryPct = min(1, max(0, (setPeak - bpm) / span))
                if bpm <= baseline + readyMargin { state = .ready; recoveryPct = 1 }
            }
        }
    }

    func label(baseline: Double) -> String {
        switch state {
        case .idle:       return "세트를 시작하세요 · 기준 \(Int(baseline)) bpm"
        case .working:    return "운동 중 — 끝나면 쉬세요"
        case .recovering:
            return recoveryPct >= 0.85
                ? String(format: "거의 회복 (%.0f%%) — 곧 다음 세트 OK", recoveryPct * 100)
                : String(format: "휴식 중 · 회복 %.0f%% — 아직 더 쉬어요", recoveryPct * 100)
        case .ready:      return "회복 완료 — 다음 세트 가도 됩니다!"
        }
    }
}
