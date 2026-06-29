import Foundation

struct CoachSignal {
    var currentHrBpm: Double
    var restingHrBpm: Double
    var confidence: Double
    var recoveryScore: Double
    var trendBpm: Double
    var completedLastSession: Bool
    var faceVisible: Bool
    var ambientLightOK: Bool
}

struct CoachDecision {
    let recommendation: String
    let intensityStep: Int
    let reason: String
}

enum CoachEngine {
    static func decide(_ signal: CoachSignal, baselineBpm: Double, confidenceThreshold: Double) -> CoachDecision {
        guard signal.faceVisible, signal.ambientLightOK, signal.confidence >= confidenceThreshold else {
            return CoachDecision(recommendation: "defer", intensityStep: 0, reason: "카메라 신호 품질이 낮거나 신뢰도가 낮음")
        }

        let hrDelta = signal.currentHrBpm - baselineBpm
        let stressed = signal.recoveryScore < 0.4 || signal.trendBpm >= 5.0 || hrDelta >= 35.0
        let stable = signal.recoveryScore >= 0.7 && abs(signal.trendBpm) <= 2.0 && hrDelta <= 20.0

        if stressed {
            return CoachDecision(recommendation: "easy", intensityStep: -1, reason: "기준 대비 심박 부하가 높거나 회복이 부족함")
        }

        if stable && signal.completedLastSession {
            return CoachDecision(recommendation: "push", intensityStep: 1, reason: "신호가 안정적이고 최근 회복이 양호함")
        }

        return CoachDecision(recommendation: "maintain", intensityStep: 0, reason: "신호는 쓸 만하지만 단계 변경엔 충분치 않음")
    }
}

