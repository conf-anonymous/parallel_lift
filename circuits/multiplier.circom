pragma circom 2.0.0;

// Simple multiplier circuit - demonstrates basic R1CS structure
// Constraint: c = a * b
template Multiplier() {
    signal input a;
    signal input b;
    signal output c;

    c <== a * b;
}

component main = Multiplier();
