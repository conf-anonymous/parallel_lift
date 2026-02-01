pragma circom 2.0.0;

// Simplified SHA256-like compression for benchmarking
// Uses bit operations simulated as field operations
// Creates a large constraint system (~5000+ constraints)

template XOR3() {
    signal input a;
    signal input b;
    signal input c;
    signal output out;

    // XOR simulation: a + b + c - 2*a*b - 2*b*c - 2*a*c + 4*a*b*c
    // For binary inputs this gives XOR
    signal ab;
    signal bc;
    signal ac;
    signal abc;

    ab <== a * b;
    bc <== b * c;
    ac <== a * c;
    abc <== ab * c;

    out <== a + b + c - 2*ab - 2*bc - 2*ac + 4*abc;
}

template AND() {
    signal input a;
    signal input b;
    signal output out;
    out <== a * b;
}

template NOT() {
    signal input in;
    signal output out;
    out <== 1 - in;
}

template MAJ() {
    signal input a;
    signal input b;
    signal input c;
    signal output out;

    // MAJ(a,b,c) = (a AND b) XOR (a AND c) XOR (b AND c)
    signal ab;
    signal ac;
    signal bc;

    ab <== a * b;
    ac <== a * c;
    bc <== b * c;

    // For binary: ab + ac + bc - 2*ab*ac - 2*ab*bc - 2*ac*bc + 4*ab*ac*bc
    // Simplified: (a*b) + (a*c) + (b*c) - 2*(a*b*c)
    signal abc;
    abc <== ab * c;
    out <== ab + ac + bc - 2*abc;
}

template CH() {
    signal input a;
    signal input b;
    signal input c;
    signal output out;

    // CH(a,b,c) = (a AND b) XOR (NOT a AND c)
    signal ab;
    signal notA;
    signal notAc;

    ab <== a * b;
    notA <== 1 - a;
    notAc <== notA * c;

    out <== ab + notAc - 2 * ab * notAc;  // XOR of ab and notAc
}

// Bit array operations for 32-bit word
template Word32Add() {
    signal input a[32];
    signal input b[32];
    signal output out[32];

    signal carry[33];
    signal sum[32];

    carry[0] <== 0;

    for (var i = 0; i < 32; i++) {
        sum[i] <== a[i] + b[i] + carry[i];

        // out[i] = sum mod 2
        out[i] <-- sum[i] % 2;
        out[i] * (1 - out[i]) === 0;

        // carry[i+1] = sum >= 2 ? 1 : 0
        carry[i+1] <-- sum[i] >= 2 ? 1 : 0;
        carry[i+1] * (1 - carry[i+1]) === 0;

        // Verify: sum = out[i] + 2 * carry[i+1]
        sum[i] === out[i] + 2 * carry[i+1];
    }
}

template SHA256Round() {
    signal input state[8][32];  // 8 words of 32 bits each
    signal input w[32];         // Message schedule word
    signal input k[32];         // Round constant
    signal output newState[8][32];

    // Compute S1 = ROTR6(e) XOR ROTR11(e) XOR ROTR25(e)
    // Simplified: just use e directly for testing
    component ch = CH();
    ch.a <== state[4][0];
    ch.b <== state[5][0];
    ch.c <== state[6][0];

    component maj = MAJ();
    maj.a <== state[0][0];
    maj.b <== state[1][0];
    maj.c <== state[2][0];

    // Simplified round: shift state and add
    component add1 = Word32Add();
    component add2 = Word32Add();

    for (var i = 0; i < 32; i++) {
        add1.a[i] <== state[7][i];
        add1.b[i] <== w[i];
    }

    for (var i = 0; i < 32; i++) {
        add2.a[i] <== add1.out[i];
        add2.b[i] <== k[i];
    }

    // State update
    for (var i = 0; i < 32; i++) {
        newState[7][i] <== state[6][i];
        newState[6][i] <== state[5][i];
        newState[5][i] <== state[4][i];
        newState[4][i] <== add2.out[i];
        newState[3][i] <== state[2][i];
        newState[2][i] <== state[1][i];
        newState[1][i] <== state[0][i];
        newState[0][i] <== add2.out[i];
    }
}

template SHA256Compress(nRounds) {
    signal input message[nRounds][32];
    signal input constants[nRounds][32];
    signal output hash[32];

    signal state[nRounds+1][8][32];

    // Initial hash values (simplified)
    for (var i = 0; i < 8; i++) {
        for (var j = 0; j < 32; j++) {
            state[0][i][j] <== (i + j) % 2;  // Simple initial state
        }
    }

    component rounds[nRounds];

    for (var r = 0; r < nRounds; r++) {
        rounds[r] = SHA256Round();
        for (var i = 0; i < 8; i++) {
            for (var j = 0; j < 32; j++) {
                rounds[r].state[i][j] <== state[r][i][j];
            }
        }
        for (var j = 0; j < 32; j++) {
            rounds[r].w[j] <== message[r][j];
            rounds[r].k[j] <== constants[r][j];
        }
        for (var i = 0; i < 8; i++) {
            for (var j = 0; j < 32; j++) {
                state[r+1][i][j] <== rounds[r].newState[i][j];
            }
        }
    }

    // Output first word as hash
    for (var j = 0; j < 32; j++) {
        hash[j] <== state[nRounds][0][j];
    }
}

// 16 rounds creates ~3000-4000 constraints
component main = SHA256Compress(16);
