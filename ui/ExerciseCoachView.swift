import SwiftUI

enum CoachRecommendation: String {
    case easy
    case maintain
    case push
    case deferMeasurement = "defer"
}

struct DemoCoachSignal {
    var currentHrBpm: Double
    var restingHrBpm: Double
    var hrConfidence: Double
    var recoveryScore: Double
    var rhrTrendBpm: Double
    var completedLastSession: Bool
    var faceVisible: Bool
    var ambientLightOK: Bool
}

struct DemoCoachDecision {
    var recommendation: CoachRecommendation
    var intensityStep: Int
    var reason: String
}

enum DemoCoachEngine {
    static func decide(_ signal: DemoCoachSignal) -> DemoCoachDecision {
        guard signal.faceVisible, signal.ambientLightOK, signal.hrConfidence >= 0.6 else {
            return DemoCoachDecision(
                recommendation: .deferMeasurement,
                intensityStep: 0,
                reason: "low-quality camera signal or low confidence"
            )
        }

        let hrDelta = signal.currentHrBpm - signal.restingHrBpm
        let stressed = signal.recoveryScore < 0.4 || signal.rhrTrendBpm >= 5.0 || hrDelta >= 35.0
        let stable = signal.recoveryScore >= 0.7 && abs(signal.rhrTrendBpm) <= 2.0 && hrDelta <= 20.0

        if stressed {
            return DemoCoachDecision(
                recommendation: .easy,
                intensityStep: -1,
                reason: "heart-rate load looks elevated relative to baseline or recovery is poor"
            )
        }

        if stable, signal.completedLastSession {
            return DemoCoachDecision(
                recommendation: .push,
                intensityStep: 1,
                reason: "signal is stable and recent recovery/completion support a small increase"
            )
        }

        return DemoCoachDecision(
            recommendation: .maintain,
            intensityStep: 0,
            reason: "signal is usable but not strong enough to justify a step change"
        )
    }
}

struct ExerciseCoachView: View {
    @State private var signal = DemoCoachSignal(
        currentHrBpm: 132,
        restingHrBpm: 68,
        hrConfidence: 0.91,
        recoveryScore: 0.78,
        rhrTrendBpm: -1.5,
        completedLastSession: true,
        faceVisible: true,
        ambientLightOK: true
    )

    private var decision: DemoCoachDecision {
        DemoCoachEngine.decide(signal)
    }

    var body: some View {
        ZStack {
            LinearGradient(
                colors: [Color(red: 0.05, green: 0.10, blue: 0.14), Color(red: 0.16, green: 0.20, blue: 0.24)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()

            VStack(spacing: 18) {
                header
                cameraCard
                metricsGrid
                recommendationCard
                safetyNote
            }
            .padding(20)
        }
    }

    private var header: some View {
        VStack(spacing: 8) {
            Text("Heartbeat Coach")
                .font(.system(size: 34, weight: .bold, design: .rounded))
                .foregroundStyle(.white)
            Text("Front camera, on-device, wellness only")
                .font(.headline)
                .foregroundStyle(.white.opacity(0.72))
        }
    }

    private var cameraCard: some View {
        ZStack(alignment: .bottomLeading) {
            RoundedRectangle(cornerRadius: 28, style: .continuous)
                .fill(Color.white.opacity(0.08))
                .overlay(
                    RoundedRectangle(cornerRadius: 28, style: .continuous)
                        .strokeBorder(Color.white.opacity(0.14), lineWidth: 1)
                )
                .frame(height: 260)

            VStack(alignment: .leading, spacing: 10) {
                Label("Camera ready", systemImage: "camera.fill")
                    .font(.headline)
                    .foregroundStyle(.white)
                Text("Hold still for a short capture. The app will reject noisy frames instead of guessing.")
                    .font(.subheadline)
                    .foregroundStyle(.white.opacity(0.75))
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(20)
        }
    }

    private var metricsGrid: some View {
        HStack(spacing: 12) {
            metricTile(title: "HR", value: "\(Int(signal.currentHrBpm)) bpm")
            metricTile(title: "Confidence", value: String(format: "%.0f%%", signal.hrConfidence * 100))
            metricTile(title: "RHR Trend", value: String(format: "%+.1f", signal.rhrTrendBpm))
        }
    }

    private func metricTile(title: String, value: String) -> some View {
        VStack(spacing: 10) {
            Text(title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.white.opacity(0.65))
            Text(value)
                .font(.headline.weight(.bold))
                .foregroundStyle(.white)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 16)
        .background(Color.white.opacity(0.08), in: RoundedRectangle(cornerRadius: 20, style: .continuous))
    }

    private var recommendationCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Recommendation")
                    .font(.headline)
                    .foregroundStyle(.white.opacity(0.75))
                Spacer()
                Text(decision.recommendation.rawValue.uppercased())
                    .font(.headline.weight(.bold))
                    .foregroundStyle(.white)
            }

            Text(decision.reason)
                .font(.body)
                .foregroundStyle(.white)

            HStack {
                Label("Intensity step \(decision.intensityStep)", systemImage: "figure.run")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.white.opacity(0.85))
                Spacer()
                Button("Retake") { }
                    .buttonStyle(.borderedProminent)
            }
        }
        .padding(18)
        .background(Color.white.opacity(0.08), in: RoundedRectangle(cornerRadius: 24, style: .continuous))
    }

    private var safetyNote: some View {
        Text("No diagnosis claims. If the frame is low quality, the coach defers instead of forcing advice.")
            .font(.footnote)
            .foregroundStyle(.white.opacity(0.6))
            .multilineTextAlignment(.center)
    }
}

#Preview {
    ExerciseCoachView()
}
