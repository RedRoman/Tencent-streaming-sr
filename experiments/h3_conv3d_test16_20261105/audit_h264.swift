// Read-only full H.264 decode audit using system macOS frameworks.
// Build: swiftc -parse-as-library audit_h264.swift -o ./audit_h264
// Run: ./audit_h264 --root REPO --output EVIDENCE.json FILE.mp4 [...]
import Foundation
import AVFoundation
import CoreVideo
import CryptoKit

struct AuditError: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}

struct Options {
    var root: URL? = nil
    var output: URL? = nil
    var expectedFrames = 120
    var expectedFPS = 24.0
    var expectedDuration = 5.0
    var paths: [String] = []
}

func parseOptions() throws -> Options {
    var result = Options()
    let args = Array(CommandLine.arguments.dropFirst())
    var i = 0
    while i < args.count {
        let arg = args[i]
        if arg == "--help" {
            print("Usage: h3_decode_audit [--root REPO] [--output JSON] [--expect-frames 120] [--expect-fps 24] [--expect-duration 5] FILE.mp4 [...]")
            exit(0)
        }
        if arg.hasPrefix("--") {
            guard i + 1 < args.count else { throw AuditError("Missing value for \(arg)") }
            i += 1
            let value = args[i]
            switch arg {
            case "--root": result.root = URL(fileURLWithPath: value).standardizedFileURL.resolvingSymlinksInPath()
            case "--output": result.output = URL(fileURLWithPath: value).standardizedFileURL
            case "--expect-frames":
                guard let n = Int(value), n > 0 else { throw AuditError("Invalid expected frame count") }
                result.expectedFrames = n
            case "--expect-fps":
                guard let x = Double(value), x.isFinite, x > 0 else { throw AuditError("Invalid expected FPS") }
                result.expectedFPS = x
            case "--expect-duration":
                guard let x = Double(value), x.isFinite, x > 0 else { throw AuditError("Invalid expected duration") }
                result.expectedDuration = x
            default: throw AuditError("Unknown option: \(arg)")
            }
        } else {
            result.paths.append(arg)
        }
        i += 1
    }
    guard !result.paths.isEmpty else { throw AuditError("No input videos") }
    return result
}

func hashFile(_ url: URL) throws -> String {
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }
    var hash = SHA256()
    while let bytes = try handle.read(upToCount: 8 << 20), !bytes.isEmpty {
        hash.update(data: bytes)
    }
    return hash.finalize().map { String(format: "%02x", $0) }.joined()
}

func timeJSON(_ t: CMTime) -> [String: Any] {
    var result: [String: Any] = ["value": t.value, "timescale": t.timescale, "epoch": t.epoch]
    let seconds = CMTimeGetSeconds(t)
    result["seconds"] = seconds.isFinite ? seconds as Any : NSNull()
    return result
}

func fourCC(_ x: FourCharCode) -> String {
    String(bytes: [UInt8((x >> 24) & 255), UInt8((x >> 16) & 255), UInt8((x >> 8) & 255), UInt8(x & 255)], encoding: .ascii) ?? "unknown"
}

func statusName(_ status: AVAssetReader.Status) -> String {
    switch status {
    case .unknown: return "unknown"
    case .reading: return "reading"
    case .completed: return "completed"
    case .failed: return "failed"
    case .cancelled: return "cancelled"
    @unknown default: return "unrecognized"
    }
}

func auditVideo(_ path: String, _ options: Options) async -> [String: Any] {
    let url = URL(fileURLWithPath: path).standardizedFileURL.resolvingSymlinksInPath()
    let decoder = "AVAssetReader / \(ProcessInfo.processInfo.operatingSystemVersionString) / decoded 32BGRA CVPixelBuffer"
    var result: [String: Any] = ["path": url.path, "decoder": decoder, "passed": false,
                               "decoded_frames": 0, "width": 0, "height": 0, "reader_status": "not_started"]
    var failures: [String] = []
    let started = Date()
    do {
        if let root = options.root {
            let prefix = root.path + "/"
            guard url.path.hasPrefix(prefix) else { throw AuditError("Input is outside --root") }
            result["path"] = String(url.path.dropFirst(prefix.count))
        }
        let before = try FileManager.default.attributesOfItem(atPath: url.path)
        let fileSize = (before[.size] as? NSNumber)?.int64Value ?? -1
        result["bytes"] = fileSize
        result["sha256"] = try hashFile(url)
        let asset = AVURLAsset(url: url)
        let tracks = try await asset.loadTracks(withMediaType: .video)
        guard tracks.count == 1, let track = tracks.first else { throw AuditError("Expected exactly one video track, found \(tracks.count)") }
        let descriptions = try await track.load(.formatDescriptions)
        let codecs = descriptions.map { fourCC(CMFormatDescriptionGetMediaSubType($0)) }
        result["codecs"] = codecs
        if codecs.isEmpty || codecs.contains(where: { $0 != "avc1" && $0 != "avc3" }) {
            failures.append("Video is not H.264/AVC")
        }
        let nominalFPS = try await track.load(.nominalFrameRate)
        result["nominal_fps"] = Double(nominalFPS)
        let trackRange = try await track.load(.timeRange)
        result["track_time_range"] = ["start": timeJSON(trackRange.start), "duration": timeJSON(trackRange.duration)]
        let assetDuration = try await asset.load(.duration)
        result["asset_duration"] = timeJSON(assetDuration)
        let reader = try AVAssetReader(asset: asset)
        // Non-nil uncompressed output settings force the decoder to produce pixels.
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else { throw AuditError("Cannot add uncompressed video track output") }
        reader.add(output)
        guard reader.startReading() else { throw reader.error ?? AuditError("startReading failed") }
        var count = 0
        var missingImages = 0
        var invalidPTS = 0
        var nonmonotonicPTS = 0
        var cadenceErrors = 0
        var sizes = Set<String>()
        var firstPTS: CMTime? = nil
        var lastPTS: CMTime? = nil
        var lastDuration: CMTime? = nil
        var maxCadenceError = 0.0
        var minDelta = Double.infinity
        var maxDelta = -Double.infinity
        var width = 0
        var height = 0
        let expectedDelta = 1.0 / options.expectedFPS
        while true {
            let consumed: Bool = autoreleasepool {
                guard let sample = output.copyNextSampleBuffer() else { return false }
                count += 1
                if let image = CMSampleBufferGetImageBuffer(sample) {
                    let w = CVPixelBufferGetWidth(image), h = CVPixelBufferGetHeight(image)
                    if count == 1 { width = w; height = h }
                    sizes.insert("\(w)x\(h)")
                } else { missingImages += 1 }
                let pts = CMSampleBufferGetPresentationTimeStamp(sample)
                let seconds = CMTimeGetSeconds(pts)
                if !pts.isNumeric || !seconds.isFinite {
                    invalidPTS += 1
                } else {
                    if firstPTS == nil { firstPTS = pts }
                    if let previous = lastPTS {
                        let delta = CMTimeGetSeconds(CMTimeSubtract(pts, previous))
                        let error = abs(delta - expectedDelta)
                        minDelta = min(minDelta, delta)
                        maxDelta = max(maxDelta, delta)
                        maxCadenceError = max(maxCadenceError, error)
                        if delta <= 0 { nonmonotonicPTS += 1 }
                        if error > 0.000001 { cadenceErrors += 1 }
                    }
                    lastPTS = pts
                }
                lastDuration = CMSampleBufferGetDuration(sample)
                return true
            }
            if !consumed { break }
        }
        result["decoded_frames"] = count
        result["width"] = width
        result["height"] = height
        result["decoded_sizes"] = sizes.sorted()
        result["missing_image_buffers"] = missingImages
        result["reader_status"] = statusName(reader.status)
        result["invalid_pts"] = invalidPTS
        result["nonmonotonic_pts"] = nonmonotonicPTS
        result["cadence_errors"] = cadenceErrors
        result["max_cadence_error_seconds"] = maxCadenceError
        if let first = firstPTS { result["first_pts"] = timeJSON(first) }
        if let last = lastPTS { result["last_pts"] = timeJSON(last) }
        if let duration = lastDuration { result["last_sample_duration"] = timeJSON(duration) }
        if minDelta.isFinite { result["min_pts_delta_seconds"] = minDelta }
        if maxDelta.isFinite { result["max_pts_delta_seconds"] = maxDelta }
        if let first = firstPTS, let last = lastPTS {
            let span = CMTimeGetSeconds(CMTimeSubtract(last, first))
            result["pts_span_seconds"] = span
            // Presentation span excludes the final frame's display duration.
            if abs(span - Double(options.expectedFrames - 1) / options.expectedFPS) > 0.000001 {
                failures.append("Decoded presentation span differs from expected cadence/count")
            }
            if abs(CMTimeGetSeconds(first)) > 0.000001 { failures.append("First decoded PTS is not zero") }
        }
        if reader.status != .completed { failures.append("Reader did not reach completed: \(reader.error?.localizedDescription ?? "no error detail")") }
        if count != options.expectedFrames { failures.append("Decoded \(count) frames; expected \(options.expectedFrames)") }
        if missingImages > 0 { failures.append("Some samples lack decoded pixel buffers") }
        if sizes.count != 1 || width < 1 || height < 1 { failures.append("Missing or changing decoded dimensions") }
        if invalidPTS + nonmonotonicPTS + cadenceErrors > 0 { failures.append("Invalid presentation timestamps or cadence") }
        if abs(Double(nominalFPS) - options.expectedFPS) > 0.0001 { failures.append("Nominal FPS differs from expected") }
        for duration in [trackRange.duration, assetDuration] {
            let seconds = CMTimeGetSeconds(duration)
            if !seconds.isFinite || abs(seconds - options.expectedDuration) > 0.001 { failures.append("Container duration differs from expected") }
        }
        let after = try FileManager.default.attributesOfItem(atPath: url.path)
        if (after[.size] as? NSNumber)?.int64Value != fileSize || (before[.modificationDate] as? Date) != (after[.modificationDate] as? Date) {
            failures.append("Input changed while hashing/decoding")
        }
    } catch {
        failures.append(String(describing: error))
    }
    result["elapsed_seconds"] = Date().timeIntervalSince(started)
    result["errors"] = failures
    result["passed"] = failures.isEmpty
    return result
}

@main struct DecodeAudit {
    static func main() async {
        do {
            let options = try parseOptions()
            var videos: [[String: Any]] = []
            for path in options.paths { videos.append(await auditVideo(path, options)) }
            let passed = videos.allSatisfy { ($0["passed"] as? Bool) == true }
            let result: [String: Any] = ["schema_version": 1, "status": passed ? "passed" : "failed", "tool": "h3_decode_audit.swift",
                                        "video_count": videos.count, "expected_frames": options.expectedFrames, "expected_fps": options.expectedFPS,
                                        "expected_duration_seconds": options.expectedDuration, "videos": videos]
            let bytes = try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
            if let output = options.output {
                let protectedInputs = Set(options.paths.map { URL(fileURLWithPath: $0).standardizedFileURL.resolvingSymlinksInPath().path })
                guard !protectedInputs.contains(output.resolvingSymlinksInPath().path) else { throw AuditError("Output path must not overwrite an input") }
                try bytes.write(to: output, options: .atomic)
            }
            FileHandle.standardOutput.write(bytes)
            FileHandle.standardOutput.write(Data([10]))
            exit(passed ? 0 : 1)
        } catch {
            FileHandle.standardError.write(Data("h3_decode_audit: \(error)\n".utf8))
            exit(2)
        }
    }
}
