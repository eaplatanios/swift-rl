import CryptoSwift
import Foundation

/// Creates a strong random seed.
public func createSeed(using seed: UInt64? = nil) -> UInt64 {
  if let s = seed {
    return UInt64(truncating: NSDecimalNumber(decimal: pow(Decimal(s % 2), 8 * 8)))
  } else {
    return UInt64.random(in: UInt64.min...UInt64.max)
  }
}

/// Creates a strong random seed.
public func createSeed(using seed: String) -> UInt64 {
  let digest = seed.sha512()
  let data = Data(bytes: digest.utf8)
  return UInt64(bigEndian: data.withUnsafeBytes { $0.pointee })
}

/// Hashes the provided seed in order to produce a stronger seed.
/// 
/// Any given evaluation is likely to have many PRNG's active at once. (Most commonly, because the 
/// environment is running in multiple processes). There is literature indicating that having linear 
/// correlations between seeds of multiple PRNG's can correlate the outputs:
///
///   - http://blogs.unity3d.com/2015/01/07/a-primer-on-repeatable-random-numbers
///   - http://stackoverflow.com/questions/1554958/how-different-do-random-seeds-need-to-be
///   - http://dl.acm.org/citation.cfm?id=1276928
/// 
/// Thus, for sanity we hash the seeds before using them. This scheme is likely not crypto-strength, 
/// but it should be good enough to get rid of simple correlations.
public func hashSeed(_ seed: UInt64) -> UInt64 {
  let bytes = [
    UInt8(truncatingIfNeeded: seed),
    UInt8(truncatingIfNeeded: seed >> (UInt8.bitWidth * 1)),
    UInt8(truncatingIfNeeded: seed >> (UInt8.bitWidth * 2)),
    UInt8(truncatingIfNeeded: seed >> (UInt8.bitWidth * 3)),
    UInt8(truncatingIfNeeded: seed >> (UInt8.bitWidth * 4)),
    UInt8(truncatingIfNeeded: seed >> (UInt8.bitWidth * 5)),
    UInt8(truncatingIfNeeded: seed >> (UInt8.bitWidth * 6)),
    UInt8(truncatingIfNeeded: seed >> (UInt8.bitWidth * 7))]
  let data = Data(bytes: bytes.sha512())
  return UInt64(bigEndian: data.withUnsafeBytes { $0.pointee })
}
