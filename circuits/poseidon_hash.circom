pragma circom 2.0.0;

// Simplified Poseidon-like hash for benchmarking
// Uses S-box (x^5) and linear layer iterations
// Creates ~500-1000 constraints depending on rounds

template SBox() {
    signal input in;
    signal output out;

    signal x2;
    signal x4;

    x2 <== in * in;
    x4 <== x2 * x2;
    out <== x4 * in;  // x^5
}

template LinearLayer(n) {
    signal input in[n];
    signal output out[n];

    // Simple MDS-like mixing
    for (var i = 0; i < n; i++) {
        var sum = 0;
        for (var j = 0; j < n; j++) {
            sum += in[j] * ((i + j + 1) % 7 + 1);
        }
        out[i] <== sum;
    }
}

template PoseidonRound(n) {
    signal input in[n];
    signal output out[n];

    component sboxes[n];
    component linear = LinearLayer(n);

    for (var i = 0; i < n; i++) {
        sboxes[i] = SBox();
        sboxes[i].in <== in[i] + (i + 1);  // Add round constant
        linear.in[i] <== sboxes[i].out;
    }

    for (var i = 0; i < n; i++) {
        out[i] <== linear.out[i];
    }
}

template PoseidonHash(nRounds) {
    signal input preimage[3];
    signal output hash;

    component rounds[nRounds];

    rounds[0] = PoseidonRound(3);
    rounds[0].in[0] <== preimage[0];
    rounds[0].in[1] <== preimage[1];
    rounds[0].in[2] <== preimage[2];

    for (var i = 1; i < nRounds; i++) {
        rounds[i] = PoseidonRound(3);
        rounds[i].in[0] <== rounds[i-1].out[0];
        rounds[i].in[1] <== rounds[i-1].out[1];
        rounds[i].in[2] <== rounds[i-1].out[2];
    }

    hash <== rounds[nRounds-1].out[0] + rounds[nRounds-1].out[1] + rounds[nRounds-1].out[2];
}

// 8 rounds creates ~100-200 constraints
component main = PoseidonHash(8);
