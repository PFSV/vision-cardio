import SwiftUI

struct ContentView: View {
    @StateObject private var vm = CoachViewModel()
    @State private var showAbout = false

    private var signal: CoachSignal { vm.signal }

    // 절제된 단일 액센트 (muted red) + 거의 검정 배경.
    private let accent = Color(red: 0.91, green: 0.30, blue: 0.34)
    private let bg = Color(red: 0.045, green: 0.05, blue: 0.06)

    private func trendSymbol(_ t: Double) -> String {
        if t > 1.5 { return "arrow.up" }
        if t < -1.5 { return "arrow.down" }
        return "arrow.right"
    }

    private func phaseColor(_ p: Int) -> Color {
        switch p {
        case 1: return Color(red: 0.40, green: 0.78, blue: 0.55)
        case 2: return Color(red: 0.55, green: 0.80, blue: 0.50)
        case 3: return Color(red: 0.86, green: 0.78, blue: 0.40)
        case 4: return Color(red: 0.90, green: 0.55, blue: 0.33)
        default: return accent
        }
    }

    private func confColor(_ c: Double) -> Color {
        if c >= 0.7 { return Color(red: 0.40, green: 0.78, blue: 0.55) }
        if c >= 0.4 { return Color(red: 0.86, green: 0.78, blue: 0.40) }
        return .white.opacity(0.35)
    }

    var body: some View {
        ZStack {
            bg.ignoresSafeArea()

            VStack(spacing: 12) {
                header

                cameraView

                heroCard

                Picker("모드", selection: $vm.mode) {
                    ForEach(ExerciseMode.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)

                modeCard

                Text(vm.statusText)
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(vm.modelLoaded ? .white.opacity(0.4) : Color(red: 0.86, green: 0.78, blue: 0.40))
                    .frame(maxWidth: .infinity, alignment: .leading)

                startButton
            }
            .padding(.horizontal, 18)
            .padding(.vertical, 10)
        }
        .tint(accent)
        .sheet(isPresented: $showAbout) {
            AboutView(accent: accent, bg: bg)
        }
    }

    // MARK: - 헤더 (군더더기 없는 투톤 워드마크만)
    private var header: some View {
        HStack(spacing: 0) {
            Text("VISION ").foregroundStyle(.white)
            Text("CARDIO").foregroundStyle(accent)
            Spacer()
            Button { showAbout = true } label: {
                Image(systemName: "info.circle")
                    .font(.system(size: 15, weight: .regular))
                    .foregroundStyle(.white.opacity(0.35))
            }
            .accessibilityLabel("정보")
        }
        .font(.system(size: 17, weight: .semibold, design: .monospaced))
        .tracking(3)
        .padding(.top, 8)
        .padding(.bottom, 2)
    }

    // MARK: - 카메라
    private var cameraView: some View {
        RoundedRectangle(cornerRadius: 18, style: .continuous)
            .fill(Color.white.opacity(0.03))
            .overlay {
                if vm.camera.isAuthorized {
                    CameraPreview(session: vm.camera.session)
                        .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
                } else {
                    VStack(spacing: 8) {
                        Image(systemName: "camera.metering.unknown")
                            .font(.system(size: 22, weight: .light))
                            .foregroundStyle(.white.opacity(0.5))
                        Text("카메라 권한이 필요합니다")
                            .font(.system(size: 13, weight: .medium))
                            .foregroundStyle(.white.opacity(0.8))
                        Text("전면 카메라 접근을 허용하세요")
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundStyle(.white.opacity(0.4))
                    }
                }
            }
            .frame(maxWidth: .infinity, minHeight: 150, maxHeight: .infinity)
            .overlay(
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .strokeBorder(.white.opacity(0.08), lineWidth: 1)
            )
    }

    // MARK: - 심박 히어로
    private var heroCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("심박").font(.system(size: 11, weight: .medium)).tracking(2)
                    .foregroundStyle(.white.opacity(0.4))
                Spacer()
                HStack(spacing: 5) {
                    Circle().fill(confColor(signal.confidence)).frame(width: 6, height: 6)
                    Text("신뢰도 \(Int(signal.confidence * 100))%")
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.5))
                }
            }
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Image(systemName: "heart.fill")
                    .font(.system(size: 26))
                    .foregroundStyle(accent)
                Text("\(Int(signal.currentHrBpm))")
                    .font(.system(size: 60, weight: .semibold, design: .monospaced))
                    .foregroundStyle(.white)
                Text("BPM").font(.system(size: 13, weight: .semibold, design: .monospaced))
                    .tracking(1).foregroundStyle(.white.opacity(0.4))
                Image(systemName: trendSymbol(signal.trendBpm))
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(.white.opacity(0.45))
                Spacer()
            }
        }
        .card(accent: accent)
    }

    // MARK: - 모드 카드
    private var modeCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            if vm.session != .active {
                Text(vm.session == .calibrating
                     ? "기준 심박 측정 중 · 가만히 계세요"
                     : "측정을 시작하면 먼저 기준 심박을 잽니다")
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(.white)
                    .fixedSize(horizontal: false, vertical: true)
                Text("정확한 기준이 있어야 페이즈·세트 판단이 정확해집니다")
                    .font(.system(size: 12)).foregroundStyle(.white.opacity(0.45))
            } else if vm.mode == .running {
                label("러닝 · 페이즈")
                Text(vm.phaseText)
                    .font(.system(size: 18, weight: .semibold)).foregroundStyle(.white)
                HStack(spacing: 5) {
                    ForEach(1...5, id: \.self) { i in
                        Capsule()
                            .fill(i <= vm.phase ? phaseColor(vm.phase) : Color.white.opacity(0.10))
                            .frame(height: 6)
                    }
                }
                Text(vm.phaseHint).font(.system(size: 12)).foregroundStyle(.white.opacity(0.55))
                baselineTag
            } else {
                label("웨이트 · 세트/휴식")
                Text(vm.weightsLabel)
                    .font(.system(size: 18, weight: .semibold)).foregroundStyle(.white)
                    .fixedSize(horizontal: false, vertical: true)
                ProgressView(value: vm.recoveryPct)
                    .tint(vm.recoveryPct >= 1 ? confColor(0.8) : accent)
                baselineTag
            }
        }
        .card(accent: accent)
    }

    private func label(_ s: String) -> some View {
        Text(s).font(.system(size: 11, weight: .medium, design: .monospaced)).tracking(1)
            .foregroundStyle(.white.opacity(0.4))
    }

    private var baselineTag: some View {
        Text("기준 휴식 \(Int(vm.baselineBpm)) bpm")
            .font(.system(size: 11, design: .monospaced))
            .foregroundStyle(.white.opacity(0.35))
    }

    // MARK: - 버튼
    private var startButton: some View {
        Button {
            if vm.camera.isRunning { vm.stop() } else { vm.start() }
        } label: {
            Text(vm.camera.isRunning ? "정지" : "측정 시작")
                .font(.system(size: 16, weight: .semibold))
                .frame(maxWidth: .infinity)
                .padding(.vertical, 15)
        }
        .foregroundStyle(vm.camera.isRunning ? .white.opacity(0.9) : .white)
        .background(
            vm.camera.isRunning ? AnyShapeStyle(.white.opacity(0.08)) : AnyShapeStyle(accent),
            in: RoundedRectangle(cornerRadius: 15, style: .continuous)
        )
        .padding(.top, 2)
    }
}

/// 공통 카드: 미세한 채움 + 헤어라인 테두리.
private struct CardStyle: ViewModifier {
    func body(content: Content) -> some View {
        content
            .padding(16)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.white.opacity(0.04), in: RoundedRectangle(cornerRadius: 18, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .strokeBorder(.white.opacity(0.08), lineWidth: 1)
            )
    }
}
private extension View {
    func card(accent: Color) -> some View { modifier(CardStyle()) }
}

// MARK: - About / 크레딧

struct AboutView: View {
    let accent: Color
    let bg: Color
    @Environment(\.dismiss) private var dismiss

    private var version: String {
        let v = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "—"
        let b = Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "—"
        return "v\(v) (\(b))"
    }

    var body: some View {
        ZStack {
            bg.ignoresSafeArea()

            VStack(alignment: .leading, spacing: 22) {
                HStack {
                    HStack(spacing: 0) {
                        Text("VISION ").foregroundStyle(.white)
                        Text("CARDIO").foregroundStyle(accent)
                    }
                    .font(.system(size: 17, weight: .semibold, design: .monospaced))
                    .tracking(3)
                    Spacer()
                    Button { dismiss() } label: {
                        Image(systemName: "xmark")
                            .font(.system(size: 14, weight: .semibold))
                            .foregroundStyle(.white.opacity(0.5))
                    }
                    .accessibilityLabel("닫기")
                }
                .padding(.top, 8)

                VStack(alignment: .leading, spacing: 8) {
                    Text("카메라로 심박수를 추정하고 운동을 코칭합니다. 모든 처리는 기기 안에서만 이루어집니다.")
                        .font(.system(size: 13))
                        .foregroundStyle(.white.opacity(0.55))
                        .fixedSize(horizontal: false, vertical: true)
                }

                Divider().overlay(.white.opacity(0.1))

                VStack(alignment: .leading, spacing: 14) {
                    creditRow(label: "제작", name: "hyeonseop yoon (PFSV)")
                    creditRow(label: "버전", name: version)
                }

                Spacer()

                Text("© 2026 hyeonseop yoon. All rights reserved.")
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.3))
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding(.bottom, 8)
            }
            .padding(.horizontal, 22)
            .padding(.vertical, 12)
        }
        .presentationDetents([.medium])
    }

    private func creditRow(label: String, name: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label)
                .font(.system(size: 11, weight: .medium, design: .monospaced))
                .tracking(1)
                .foregroundStyle(.white.opacity(0.4))
                .frame(width: 52, alignment: .leading)
            Text(name)
                .font(.system(size: 15, weight: .medium))
                .foregroundStyle(.white)
            Spacer()
        }
    }
}

#Preview {
    ContentView()
}

#Preview("About") {
    AboutView(accent: Color(red: 0.91, green: 0.30, blue: 0.34),
              bg: Color(red: 0.045, green: 0.05, blue: 0.06))
}
