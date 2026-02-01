pragma circom 2.0.0;

// Range proof circuit - proves a value is within [0, 2^n)
// Uses bit decomposition approach
// Creates n constraints for n-bit range

template Num2Bits(n) {
    signal input in;
    signal output out[n];

    var lc = 0;
    var e2 = 1;

    for (var i = 0; i < n; i++) {
        out[i] <-- (in >> i) & 1;
        out[i] * (out[i] - 1) === 0;  // Binary constraint
        lc += out[i] * e2;
        e2 = e2 * 2;
    }

    lc === in;  // Reconstruction constraint
}

template RangeProof(n) {
    signal input value;
    signal input maxValue;  // Not used directly, just for documentation

    component bits = Num2Bits(n);
    bits.in <== value;

    // All bits are binary (enforced in Num2Bits)
    // Value reconstructs correctly (enforced in Num2Bits)
    // Therefore: 0 <= value < 2^n
}

// 64-bit range proof - creates ~64 constraints
component main = RangeProof(64);
