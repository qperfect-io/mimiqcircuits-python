// Quantum Experience (QE) Standard Header
// file: qelib1.inc
// modified to be used by MIMIQ

// --- QE Hardware primitives ---

// 3-parameter 2-pulse single qubit gate
opaque u3(theta,phi,lambda) q;
// 2-parameter 1-pulse single qubit gate
opaque u2(phi,lambda) q;
// 1-parameter 0-pulse single qubit gate
opaque u1(lambda) q;
// controlled-NOT
opaque cx c,t;
// idle gate (identity)
opaque id a;
// idle gate (identity) with length gamma*sqglen
gate u0(gamma) q { U(0,0,0) q; }

// --- QE Standard Gates ---

// Pauli gate: bit-flip
opaque x a;
// Pauli gate: bit and phase flip
opaque y a;
// Pauli gate: phase flip
opaque z a;
// Clifford gate: Hadamard
opaque h a;
// Clifford gate: sqrt(Z) phase gate
opaque s a;
// Clifford gate: conjugate of sqrt(Z)
opaque sdg a;
// C3 gate: sqrt(S) phase gate
opaque t a;
// C3 gate: conjugate of sqrt(S)
opaque tdg a;

// --- Standard rotations ---
// Rotation around X-axis
opaque rx(theta) a;
// rotation around Y-axis
opaque ry(theta) a;
// rotation around Z axis
opaque rz(phi) a;

// --- QE Standard User-Defined Gates  ---

// controlled-Phase
opaque cz a,b;
// controlled-Y
opaque cy a,b;
// swap
opaque swap a,b;
// controlled-H
opaque ch a,b;
// C3 gate: Toffoli
opaque ccx a,b,c;

// cswap (Fredkin)
opaque cswap a,b,c;

// controlled rx rotation
opaque crx(lambda) a,b;

// controlled ry rotation
opaque cry(lambda) a,b;

// controlled rz rotation
opaque crz(lambda) a,b;

// controlled phase rotation
opaque cu1(lambda) a,b;

// controlled-U
opaque cu3(theta,phi,lambda) c, t;

// two-qubit XX rotation
opaque rxx(theta) a,b;

// two-qubit ZZ rotation
opaque rzz(theta) a,b;

// relative-phase CCX
gate rccx a,b,c
{
  u2(0,pi) c;
  u1(pi/4) c;
  cx b, c;
  u1(-pi/4) c;
  cx a, c;
  u1(pi/4) c;
  cx b, c;
  u1(-pi/4) c;
  u2(0,pi) c;
}

// relative-phase 3-controlled X gate
gate rc3x a,b,c,d
{
  u2(0,pi) d;
  u1(pi/4) d;
  cx c,d;
  u1(-pi/4) d;
  u2(0,pi) d;
  cx a,d;
  u1(pi/4) d;
  cx b,d;
  u1(-pi/4) d;
  cx a,d;
  u1(pi/4) d;
  cx b,d;
  u1(-pi/4) d;
  u2(0,pi) d;
  u1(pi/4) d;
  cx c,d;
  u1(-pi/4) d;
  u2(0,pi) d;
}

// 3-controlled X gate
opaque c3x a,b,c,d;

// 3-controlled sqrt(X) gate, this equals the C3X gate where the CU1 rotations are -pi/8 not -pi/4
opaque c3sqrtx a,b,c,d;

// 4-controlled X gate
opaque c4x a,b,c,d,e;
