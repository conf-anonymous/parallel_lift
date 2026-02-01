import BigInt
import Foundation

/// RNS (Residue Number System) context for FHE-style operations
/// Manages a set of coprime moduli and provides basis conversion operations
struct RNSContext {
    /// The moduli defining the RNS basis (typically 30-60 bit primes)
    let moduli: [UInt64]

    /// Product of all moduli (for CRT reconstruction)
    let modulusProduct: BigInt

    /// Precomputed values for fast basis conversion
    /// puncturedProducts[i] = M / moduli[i] where M = product of all moduli
    let puncturedProducts: [BigInt]

    /// puncturedProductsMod[i][j] = puncturedProducts[i] mod moduli[j]
    let puncturedProductsMod: [[UInt64]]

    /// Modular inverses: invPunctured[i] = (M/moduli[i])^(-1) mod moduli[i]
    let invPunctured: [UInt64]

    /// Number of moduli in this basis
    var count: Int { moduli.count }

    init(moduli: [UInt64]) {
        self.moduli = moduli

        // Compute product M
        var M = BigInt(1)
        for m in moduli {
            M *= BigInt(m)
        }
        self.modulusProduct = M

        // Compute punctured products M_i = M / q_i
        var punctured: [BigInt] = []
        for m in moduli {
            punctured.append(M / BigInt(m))
        }
        self.puncturedProducts = punctured

        // Compute punctured products mod each modulus
        var puncturedMod: [[UInt64]] = []
        for i in 0..<moduli.count {
            var row: [UInt64] = []
            for j in 0..<moduli.count {
                let val = punctured[i] % BigInt(moduli[j])
                row.append(UInt64(val))
            }
            puncturedMod.append(row)
        }
        self.puncturedProductsMod = puncturedMod

        // Compute modular inverses
        var invs: [UInt64] = []
        for i in 0..<moduli.count {
            let Mi = punctured[i] % BigInt(moduli[i])
            let inv = RNSContext.modInverse(UInt64(Mi), moduli[i])
            invs.append(inv)
        }
        self.invPunctured = invs
    }

    /// Generate a typical FHE-style RNS context with k moduli of approximately `bits` bits each
    static func generate(count k: Int, bits: Int = 60) -> RNSContext {
        // Generate k primes of approximately `bits` bits
        // For FHE, these are typically NTT-friendly primes of form p = 1 (mod 2N)
        // For our exploration, we use general primes
        var primes: [UInt64] = []
        var candidate: UInt64 = (1 << (bits - 1)) + 1

        while primes.count < k {
            if RNSContext.isPrime(candidate) {
                primes.append(candidate)
            }
            candidate += 2
        }

        return RNSContext(moduli: primes)
    }

    /// Simple primality test
    static func isPrime(_ n: UInt64) -> Bool {
        if n < 2 { return false }
        if n == 2 { return true }
        if n % 2 == 0 { return false }

        var d = n - 1
        var r = 0
        while d % 2 == 0 {
            d /= 2
            r += 1
        }

        // Miller-Rabin with small bases
        let witnesses: [UInt64] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses {
            if a >= n { continue }
            if !RNSContext.millerRabinTest(n, a, d, r) {
                return false
            }
        }
        return true
    }

    private static func millerRabinTest(_ n: UInt64, _ a: UInt64, _ d: UInt64, _ r: Int) -> Bool {
        var x = modPow(a, d, n)
        if x == 1 || x == n - 1 { return true }

        for _ in 0..<(r - 1) {
            x = modMul(x, x, n)
            if x == n - 1 { return true }
        }
        return false
    }

    /// Modular exponentiation
    static func modPow(_ base: UInt64, _ exp: UInt64, _ mod: UInt64) -> UInt64 {
        var result: UInt64 = 1
        var b = base % mod
        var e = exp

        while e > 0 {
            if e & 1 == 1 {
                result = modMul(result, b, mod)
            }
            e >>= 1
            b = modMul(b, b, mod)
        }
        return result
    }

    /// Modular multiplication avoiding overflow
    static func modMul(_ a: UInt64, _ b: UInt64, _ mod: UInt64) -> UInt64 {
        // Use 128-bit arithmetic
        let product = UInt128(a) * UInt128(b)
        return (product % UInt128(mod)).toUInt64()
    }

    /// Extended GCD for modular inverse
    static func modInverse(_ a: UInt64, _ m: UInt64) -> UInt64 {
        var t: Int64 = 0
        var newt: Int64 = 1
        var r: Int64 = Int64(m)
        var newr: Int64 = Int64(a)

        while newr != 0 {
            let quotient = r / newr
            (t, newt) = (newt, t - quotient * newt)
            (r, newr) = (newr, r - quotient * newr)
        }

        if t < 0 { t += Int64(m) }
        return UInt64(t)
    }
}

/// 128-bit unsigned integer for intermediate calculations
struct UInt128 {
    var high: UInt64
    var low: UInt64

    init(_ value: UInt64) {
        self.high = 0
        self.low = value
    }

    init(high: UInt64, low: UInt64) {
        self.high = high
        self.low = low
    }

    static func * (lhs: UInt128, rhs: UInt128) -> UInt128 {
        // For simplicity, handle 64x64 -> 128
        let a = lhs.low
        let b = rhs.low

        let aLo = a & 0xFFFFFFFF
        let aHi = a >> 32
        let bLo = b & 0xFFFFFFFF
        let bHi = b >> 32

        let p0 = aLo * bLo
        let p1 = aLo * bHi
        let p2 = aHi * bLo
        let p3 = aHi * bHi

        let mid = p1 + p2
        let midCarry: UInt64 = mid < p1 ? 1 : 0

        let low = p0 + (mid << 32)
        let lowCarry: UInt64 = low < p0 ? 1 : 0

        let high = p3 + (mid >> 32) + (midCarry << 32) + lowCarry

        return UInt128(high: high, low: low)
    }

    static func % (lhs: UInt128, rhs: UInt128) -> UInt128 {
        // Simple modulo for 128-bit by 64-bit
        if lhs.high == 0 {
            return UInt128(lhs.low % rhs.low)
        }

        // Use BigInt for accuracy
        let lhsBig = BigInt(lhs.high) << 64 + BigInt(lhs.low)
        let rhsBig = BigInt(rhs.low)
        let result = lhsBig % rhsBig
        return UInt128(UInt64(result))
    }

    func toUInt64() -> UInt64 {
        return low
    }
}
