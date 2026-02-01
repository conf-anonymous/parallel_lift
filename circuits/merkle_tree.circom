pragma circom 2.0.0;

// Merkle tree membership proof circuit
// Creates ~1000-2000 constraints for depth 10

template HashPair() {
    signal input left;
    signal input right;
    signal output hash;

    // Simplified hash: (left^2 + right^2) * (left + right) + left*right
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
    signal input pathBit;  // 0 or 1
    signal output hashLeft;
    signal output hashRight;

    // If pathBit == 0: (left, right), else (right, left)
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

    // Ensure pathBit is binary
    pathBit * (1 - pathBit) === 0;
}

template MerkleProof(depth) {
    signal input leaf;
    signal input root;
    signal input siblings[depth];
    signal input pathBits[depth];

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

    // Verify computed root matches expected root
    root === levels[depth-1].parent;
}

// Depth 10 Merkle tree (supports 1024 leaves)
// Creates ~300-400 constraints
component main {public [root]} = MerkleProof(10);
