pragma circom 2.0.0;

// Batch Merkle proof verification - larger circuit for benchmarking
// Verifies multiple Merkle proofs in a single circuit
// Creates ~2000-3000 constraints

template HashPair() {
    signal input left;
    signal input right;
    signal output hash;

    signal leftSq;
    signal rightSq;
    signal sum;
    signal prod;
    signal sumSq;

    leftSq <== left * left;
    rightSq <== right * right;
    sum <== left + right;
    prod <== left * right;
    sumSq <== (leftSq + rightSq) * sum;
    hash <== sumSq + prod;
}

template MerkleSelector() {
    signal input left;
    signal input right;
    signal input pathBit;
    signal output hashLeft;
    signal output hashRight;

    hashLeft <== left + pathBit * (right - left);
    hashRight <== right + pathBit * (left - right);
}

template MerkleLevel() {
    signal input leaf;
    signal input sibling;
    signal input pathBit;
    signal output parent;

    component selector = MerkleSelector();
    component hasher = HashPair();

    selector.left <== leaf;
    selector.right <== sibling;
    selector.pathBit <== pathBit;

    hasher.left <== selector.hashLeft;
    hasher.right <== selector.hashRight;

    parent <== hasher.hash;

    pathBit * (1 - pathBit) === 0;
}

template SingleMerkleProof(depth) {
    signal input leaf;
    signal input root;
    signal input siblings[depth];
    signal input pathBits[depth];
    signal output verified;

    component levels[depth];

    levels[0] = MerkleLevel();
    levels[0].leaf <== leaf;
    levels[0].sibling <== siblings[0];
    levels[0].pathBit <== pathBits[0];

    for (var i = 1; i < depth; i++) {
        levels[i] = MerkleLevel();
        levels[i].leaf <== levels[i-1].parent;
        levels[i].sibling <== siblings[i];
        levels[i].pathBit <== pathBits[i];
    }

    // Output 1 if verified, needs to equal root
    signal diff;
    diff <== levels[depth-1].parent - root;
    verified <== 1 - diff * diff;  // Will be 1 only if diff == 0
}

template BatchMerkleProof(nProofs, depth) {
    signal input leaves[nProofs];
    signal input root;
    signal input siblings[nProofs][depth];
    signal input pathBits[nProofs][depth];
    signal output allVerified;

    component proofs[nProofs];
    signal verifiedAcc[nProofs + 1];
    verifiedAcc[0] <== 1;

    for (var i = 0; i < nProofs; i++) {
        proofs[i] = SingleMerkleProof(depth);
        proofs[i].leaf <== leaves[i];
        proofs[i].root <== root;
        for (var j = 0; j < depth; j++) {
            proofs[i].siblings[j] <== siblings[i][j];
            proofs[i].pathBits[j] <== pathBits[i][j];
        }
        // Accumulate verification results
        verifiedAcc[i+1] <== verifiedAcc[i] * proofs[i].verified;
    }

    allVerified <== verifiedAcc[nProofs];
}

// 8 proofs of depth 12 each - creates ~1500-2000 constraints
component main {public [root]} = BatchMerkleProof(8, 12);
