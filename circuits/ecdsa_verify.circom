pragma circom 2.0.0;

// Simplified ECDSA-like signature verification for benchmarking
// Real ECDSA would use actual elliptic curve operations
// This creates a larger constraint system for stress testing

template FieldMul() {
    signal input a;
    signal input b;
    signal output c;
    c <== a * b;
}

// Simulate scalar multiplication with repeated operations
// Uses array signals to avoid loop declaration issues
template ScalarMulSimulator(n) {
    signal input scalar;
    signal input baseX;
    signal input baseY;
    signal output outX;
    signal output outY;

    signal accX[n+1];
    signal accY[n+1];
    signal bits[n];
    signal dblX[n];
    signal dblY[n];

    accX[0] <== baseX;
    accY[0] <== baseY;

    // Bit decomposition of scalar
    var lc = 0;
    var e2 = 1;
    for (var i = 0; i < n; i++) {
        bits[i] <-- (scalar >> i) & 1;
        bits[i] * (bits[i] - 1) === 0;
        lc += bits[i] * e2;
        e2 *= 2;
    }
    lc === scalar;

    // Simulated point operations (simplified, not real EC math)
    for (var i = 0; i < n; i++) {
        // Double
        dblX[i] <== accX[i] * accX[i] + 3;  // Simplified doubling formula
        dblY[i] <== accY[i] * accY[i] + accX[i];

        // Conditional add (if bit is 1)
        accX[i+1] <== dblX[i] + bits[i] * baseX;
        accY[i+1] <== dblY[i] + bits[i] * baseY;
    }

    outX <== accX[n];
    outY <== accY[n];
}

template ECDSAVerifySimulator() {
    signal input msgHash;
    signal input pubKeyX;
    signal input pubKeyY;
    signal input sigR;
    signal input sigS;
    signal input sInvHint;  // Witness hint for s^-1

    // Verify s^-1 is correct
    signal sInvCheck;
    sInvCheck <== sInvHint * sigS;
    sInvCheck === 1;

    // u1 = hash * s^-1 mod n
    signal u1;
    u1 <== msgHash * sInvHint;

    // u2 = r * s^-1 mod n
    signal u2;
    u2 <== sigR * sInvHint;

    // Simulate u1*G + u2*P
    component scalarMul1 = ScalarMulSimulator(32);
    scalarMul1.scalar <== u1;
    scalarMul1.baseX <== 1;  // Generator point G
    scalarMul1.baseY <== 2;

    component scalarMul2 = ScalarMulSimulator(32);
    scalarMul2.scalar <== u2;
    scalarMul2.baseX <== pubKeyX;
    scalarMul2.baseY <== pubKeyY;

    // Combined point (simplified addition)
    signal combinedX;
    signal combinedY;
    combinedX <== scalarMul1.outX + scalarMul2.outX;
    combinedY <== scalarMul1.outY + scalarMul2.outY;

    // Verify r == combinedX mod n
    sigR === combinedX;
}

// Creates ~500 constraints
component main {public [msgHash]} = ECDSAVerifySimulator();
